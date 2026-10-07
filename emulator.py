'''Modified from PhotoMultiEmulator
'''
import os, sys
from functools import partial
import numpy as np
import jax

from prospect.models import ProspectorParams, priors
from prospect.models import priors_beta as PZ
from utils import get_dir, piratedir

multiemul_file = os.path.join(get_dir(dirtype='pirate', outdir=piratedir), "parrot_v4_obsphot_512n_5l_24s_00z24.npy")


# Compiled photometry prediction. Running the same JAX operations eagerly costs ~1 ms per
# call (each operation is dispatched separately); compiled, they run as one XLA program.
# All parrot sub-networks have the same shapes, so this is traced once per number of
# contributing sub-networks (1, or 2 in a redshift overlap) and then reused.
@partial(jax.jit, static_argnames='act')
def _predict_phot_jit(theta, coefs, nets, act):
    result = 0.0
    for coef, (norm, denorm, wght, bias) in zip(coefs, nets):
        # Normalize layer
        x = (theta - norm[0]) / norm[1]
        # Nonlinear layers
        for i in range(len(wght) - 1):
            x = act(jax.numpy.dot(wght[i], x) + bias[i])
        # Linear layer, then denormalize
        x = jax.numpy.dot(wght[-1], x) + bias[-1]
        result = result + coef * ((denorm[1] * x) + denorm[0])
    # AB magnitudes to maggies (see EmulatorBeta.demagify)
    return jax.numpy.exp(jax.numpy.multiply(result, -0.4) / jax.numpy.log10(jax.numpy.e))


class EmulatorBeta(ProspectorParams):
    '''works with the p-beta priors:
    '''

    def __init__(self, model_params, fp=multiemul_file, obs=None, param_order=None):
        super(EmulatorBeta, self).__init__(model_params, param_order=param_order)

        # prior bounds, used by prior_product
        self._prior_lo, self._prior_hi = self._free_prior_bounds()

        # Load all emulator data.
        self.dat = np.load(fp, allow_pickle=True).all()

        #print("emulator fp =", fp)
        #print("type self.dat =", type(self.dat))
        #print("self.dat =", self.dat)

        # Determine the indices of the filters in obs['filters']
        self.sorter = np.array([self.dat["filter_redir"][f.name] for f in obs["filters"]], dtype=int)
        ### change to run both prosp_oldjax and original prosp???
        #if isinstance(self.dat, np.ndarray) and self.dat.ndim == 0:
        #    self.dat = self.dat.item()
        #filter_redir = self.dat["filter_redir"]
        #if isinstance(filter_redir, np.ndarray) and filter_redir.ndim == 0:
        #    filter_redir = filter_redir.item()
        #self.sorter = np.array([filter_redir[f.name] for f in obs["filters"]], dtype=int)

        # TODO: Check if prior bounds are with emulbounds.
        # This will require messing with the massmet prior (which uses prior.bounds()) to
        # bring it in line with the other priors (which use prior.a and prior.b)

        # TODO: Use dat["resid_quants"] data to modify obs["maggies_unc"]

        # needed for re-ordering thetas
        self.zred_index = self.dat["zred_index"]

        # Convert ANN things to JAX arrays.
        self.nets = self.dat["nets"]
        for (i, net) in enumerate(self.nets):
            # Unpack this sub-network
            norm, denorm, wght, bias, act = net

            # Translate things into JAX and remove irrelevant filters
            norm = [jax.numpy.array(n) for n in norm]
            denorm = [jax.numpy.array(d[self.sorter]) for d in denorm]
            wght = [jax.numpy.array(W) for W in wght]
            bias = [jax.numpy.array(b) for b in bias]
            wght[-1] = wght[-1][self.sorter, :]
            bias[-1] = bias[-1][self.sorter]

            # The activation function is stored as a string - this converts it
            # to the JAX function. Assumes that all nonlinear layers use the same
            # activation function in a given sub-network and that the last layer
            # is linear activation.
            act = getattr(jax.nn, act)

            # Repack
            self.nets[i] = (norm, denorm, wght, bias, act)

    # Split zredmassmet and put each into the correct index
    def modify_theta(self, theta):
        new_theta = theta[1:]
        new_theta = np.insert(new_theta, self.zred_index, theta[0])
        return new_theta

    # In nested sampling the prior enters through prior_transform, so prospector's
    # lnprobfn uses prior_product(nested=True) only as a check that theta is inside the
    # prior (it returns 0 or -inf), but it evaluates the full joint prior to do so on
    # every likelihood call (~0.3 ms). The check can only fail when a parameter sits on
    # or beyond a prior bound (in practice: transforms of unit-cube values within ~1e-16
    # of 0 or 1). So return 0 directly when every parameter is strictly inside its
    # bounds, and otherwise run the full check as before. The full prior (nested=False,
    # e.g. in write_hdf5) is unchanged.
    def prior_product(self, theta, nested=False, **extras):
        if nested and self._prior_lo is not None:
            if np.all((theta > self._prior_lo) & (theta < self._prior_hi)):
                return 0.0
        return super(EmulatorBeta, self).prior_product(theta, nested=nested, **extras)

    # Lower/upper bounds of every free parameter, for prior_product. None (always run
    # the full check) if any prior is of a type whose bounds aren't known here.
    def _free_prior_bounds(self):
        lo, hi = np.zeros(self.ndim), np.zeros(self.ndim)
        for k, inds in self.theta_index.items():
            p = self.config_dict[k]['prior']
            n = len(range(self.ndim)[inds])
            if isinstance(p, PZ.PhiSFH):
                r = p.range # zred, logmass, logzsol, logsfr_ratios
                b = list(r[:3]) + [r[3]] * (n - 3)
            elif isinstance(p, (priors.FastUniform, priors.FastTruncatedNormal)):
                b = [(p.a, p.b)] * n
            else:
                return None, None
            lo[inds], hi[inds] = np.array(b, dtype=float).T
        return lo, hi

    # Determine the relevant angle needed for the sin and cos functions in
    # transition_coef. z0 is the center of an sub-network overlap and deltaz
    # is the full width of the overlap.
    def transition_theta(self, z, z0, deltaz):
        return (np.pi / (2 * deltaz)) * (z - z0 + 0.5*deltaz)

    # Compute how much each sub-network should contribute to the overall
    # prediction. Equal to 1 where the sub-network is the only one trained,
    # 0 where the sub-emulator is not trained, and between 0 and 1 in overlaps.
    # (plain numpy: these are scalars, and eager jax calls cost ~10 us each)
    def transition_coef(self, z, z0l, deltazl, z0r, deltazr):
        if (z <= (z0l - 0.5*deltazl)) or (z >= (z0r + 0.5*deltazr)):
            return 0.0
        elif (z >= (z0l + 0.5*deltazl)) and (z <= (z0r - 0.5*deltazr)):
            return 1.0
        elif (z > (z0l - 0.5*deltazl)) and (z < (z0l + 0.5*deltazl)):
            return np.sin(self.transition_theta(z, z0l, deltazl))**2
        else:
            return np.cos(self.transition_theta(z, z0r, deltazr))**2

    # Run a forward pass through a sub-network.
    def forward_pass(self, in_vec, net):
        # Grab NN stuff
        norm, denorm, wght, bias, act = net

        # Normalize layer
        result = (in_vec - norm[0]) / norm[1]

        # Nonlinear layers
        for i in range(len(wght) - 1):
            result = act(jax.numpy.dot(wght[i], result) + bias[i])

        # Linear layer.
        result = jax.numpy.dot(wght[-1], result) + bias[-1]

        # Denormalize layer.
        result = (denorm[1] * result) + denorm[0]

        return result

    # Convert AB magnitudes to maggies.
    #
    # Note: The emulator actually outputs arsinh magnitudes, but here we
    # are assuming that these arsinh mags are exactly equal to their
    # logarithmic AB counterparts. This is a very safe assumption for AB
    # magnitudes brighter than about +33 (they asymptotically become equal
    # as the AB magnitude gets brighter), but becomes very poor fainter than
    # +34.
    def demagify(self, mag):
        result = jax.numpy.multiply(mag, -0.4)
        result = jax.numpy.divide(result, jax.numpy.log10(jax.numpy.e))
        result = jax.numpy.exp(result)

        return result

    # Use the full emulator suite to predict a set of photometry for a given input
    # theta_in parameter vector.
    def predict_phot(self, theta_in, **extras):
        # Add redshift to theta if needed.
        theta = self.modify_theta(theta_in)

        # Grab the redshift from theta.
        z = theta[self.zred_index]

        # Compute the coefficients to multiply each emulator by.
        coefs = [self.transition_coef(z, *emul_lim) for emul_lim in self.dat["emulator_limits"]]

        # Only sub-networks with a nonzero coefficient contribute.
        active = [i for (i, coef) in enumerate(coefs) if coef != 0.0]
        if len(active) == 0:
            return self.demagify(jax.numpy.zeros(len(self.sorter)))

        # Evaluate them, sum, and convert to maggies, in one compiled call.
        return _predict_phot_jit(theta, tuple(coefs[i] for i in active),
                                 tuple(self.nets[i][:4] for i in active),
                                 act=self.nets[active[0]][4])

    # Run self.predict_phot but return the result as a NumPy array
    def predict_phot_np(self, theta_in, **extras):
        result = self.predict_phot(theta_in)
        result = np.array(result)
        return result

    # We don't predict spectra, both because we don't have an emulator available
    # and because it would take too long anyway. Thus, let's just generate a vector
    # of zeros and return that in case Prospector complains about not having a
    # spectrum predicted.
    def predict_spec(self, theta, obs=None, **extras):
        return np.zeros(5994)

    # Same thing with mfrac, although we *do* have an emulator for this.
    def predict_mfrac(self, theta, obs=None, **extras):
        return -1.0

    # General predict function, returns useless mfrac and spec right now.
    def predict(self, theta, obs=None, **extras):
        return (self.predict_spec(theta, obs=obs, **extras),
                self.predict_phot_np(theta, obs=obs, **extras),
                self.predict_mfrac(theta, obs=obs, **extras))


class EmulatorDefault(ProspectorParams):
    '''works with the full 18 params default settings (p-alpha: MassMet prior, flat SFH)
    '''

    def __init__(self, model_params, fp=multiemul_file, obs=None, param_order=None):
        super(EmulatorDefault, self).__init__(model_params, param_order=param_order)

        # Load all emulator data.
        self.dat = np.load(fp, allow_pickle=True).all()

        # Determine the indices of the filters in obs['filters']
        self.sorter = np.array([self.dat["filter_redir"][f.name] for f in obs["filters"]], dtype=int)

        # TODO: Check if prior bounds are with emulbounds.
        # This will require messing with the massmet prior (which uses prior.bounds()) to
        # bring it in line with the other priors (which use prior.a and prior.b)

        # TODO: Use dat["resid_quants"] data to modify obs["maggies_unc"]

        # needed for re-ordering thetas
        self.zred_index = self.dat["zred_index"]

        # Convert ANN things to JAX arrays.
        self.nets = self.dat["nets"]
        for (i, net) in enumerate(self.nets):
            # Unpack this sub-network
            norm, denorm, wght, bias, act = net

            # Translate things into JAX and remove irrelevant filters
            norm = [jax.numpy.array(n) for n in norm]
            denorm = [jax.numpy.array(d[self.sorter]) for d in denorm]
            wght = [jax.numpy.array(W) for W in wght]
            bias = [jax.numpy.array(b) for b in bias]
            wght[-1] = wght[-1][self.sorter, :]
            bias[-1] = bias[-1][self.sorter]

            # The activation function is stored as a string - this converts it
            # to the JAX function. Assumes that all nonlinear layers use the same
            # activation function in a given sub-network and that the last layer
            # is linear activation.
            act = getattr(jax.nn, act)

            # Repack
            self.nets[i] = (norm, denorm, wght, bias, act)

    # No modifications needed since we use the full 18 params and default priors
    def modify_theta(self, theta):
        return theta

    # Determine the relevant angle needed for the sin and cos functions in
    # transition_coef. z0 is the center of an sub-network overlap and deltaz
    # is the full width of the overlap.
    def transition_theta(self, z, z0, deltaz):
        return (np.pi / (2 * deltaz)) * (z - z0 + 0.5*deltaz)

    # Compute how much each sub-network should contribute to the overall
    # prediction. Equal to 1 where the sub-network is the only one trained,
    # 0 where the sub-emulator is not trained, and between 0 and 1 in overlaps.
    # (plain numpy: these are scalars, and eager jax calls cost ~10 us each)
    def transition_coef(self, z, z0l, deltazl, z0r, deltazr):
        if (z <= (z0l - 0.5*deltazl)) or (z >= (z0r + 0.5*deltazr)):
            return 0.0
        elif (z >= (z0l + 0.5*deltazl)) and (z <= (z0r - 0.5*deltazr)):
            return 1.0
        elif (z > (z0l - 0.5*deltazl)) and (z < (z0l + 0.5*deltazl)):
            return np.sin(self.transition_theta(z, z0l, deltazl))**2
        else:
            return np.cos(self.transition_theta(z, z0r, deltazr))**2

    # Run a forward pass through a sub-network.
    def forward_pass(self, in_vec, net):
        # Grab NN stuff
        norm, denorm, wght, bias, act = net

        # Normalize layer
        result = (in_vec - norm[0]) / norm[1]

        # Nonlinear layers
        for i in range(len(wght) - 1):
            result = act(jax.numpy.dot(wght[i], result) + bias[i])

        # Linear layer.
        result = jax.numpy.dot(wght[-1], result) + bias[-1]

        # Denormalize layer.
        result = (denorm[1] * result) + denorm[0]

        return result

    # Convert AB magnitudes to maggies.
    #
    # Note: The emulator actually outputs arsinh magnitudes, but here we
    # are assuming that these arsinh mags are exactly equal to their
    # logarithmic AB counterparts. This is a very safe assumption for AB
    # magnitudes brighter than about +33 (they asymptotically become equal
    # as the AB magnitude gets brighter), but becomes very poor fainter than
    # +34.
    def demagify(self, mag):
        result = jax.numpy.multiply(mag, -0.4)
        result = jax.numpy.divide(result, jax.numpy.log10(jax.numpy.e))
        result = jax.numpy.exp(result)

        return result

    # Use the full emulator suite to predict a set of photometry for a given input
    # theta_in parameter vector.
    def predict_phot(self, theta_in, **extras):
        # returns theta_in
        # No modifications needed since we use the full 18 params and default priors
        theta = self.modify_theta(theta_in)

        # Grab the redshift from theta.
        z = theta[self.zred_index]

        # Compute the coefficients to multiply each emulator by.
        coefs = [self.transition_coef(z, *emul_lim) for emul_lim in self.dat["emulator_limits"]]

        # Only sub-networks with a nonzero coefficient contribute.
        active = [i for (i, coef) in enumerate(coefs) if coef != 0.0]
        if len(active) == 0:
            return self.demagify(jax.numpy.zeros(len(self.sorter)))

        # Evaluate them, sum, and convert to maggies, in one compiled call.
        return _predict_phot_jit(theta, tuple(coefs[i] for i in active),
                                 tuple(self.nets[i][:4] for i in active),
                                 act=self.nets[active[0]][4])

    # Run self.predict_phot but return the result as a NumPy array
    def predict_phot_np(self, theta_in, **extras):
        result = self.predict_phot(theta_in)
        result = np.array(result)
        return result

    # We don't predict spectra, both because we don't have an emulator available
    # and because it would take too long anyway. Thus, let's just generate a vector
    # of zeros and return that in case Prospector complains about not having a
    # spectrum predicted.
    def predict_spec(self, theta, obs=None, **extras):
        return np.zeros(5994)

    # Same thing with mfrac, although we *do* have an emulator for this.
    def predict_mfrac(self, theta, obs=None, **extras):
        return -1.0

    # General predict function, returns useless mfrac and spec right now.
    def predict(self, theta, obs=None, **extras):
        return (self.predict_spec(theta, obs=obs, **extras),
                self.predict_phot_np(theta, obs=obs, **extras),
                self.predict_mfrac(theta, obs=obs, **extras))
