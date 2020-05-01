package hex.ensemble;

import hex.Model;
import hex.ModelBuilder;
import hex.ModelCategory;
import hex.ensemble.Metalearner.Algorithm;
import hex.glm.GLM;
import hex.glm.GLMModel;
import hex.glm.GLMModel.GLMParameters;
import water.exceptions.H2OIllegalArgumentException;
import water.nbhm.NonBlockingHashMap;
import water.util.Log;

import java.util.ServiceLoader;
import java.util.function.Supplier;

/**
 * Entry point class to load and access the supported metalearners.
 * Most of them are defined in this class, but some others can be loaded dynamically from the classpath,
 * this is for example the case with the XGBoostMetalearner.
 */
public class Metalearners {

    static final NonBlockingHashMap<String, MetalearnerProvider> providersByName = new NonBlockingHashMap<>();

    static {
        LocalProvider[] localProviders = new LocalProvider[] {
                new LocalProvider<>(Algorithm.AUTO, AUTOMetalearner::new),
                new LocalProvider<>(Algorithm.deeplearning, DLMetalearner::new),
                new LocalProvider<>(Algorithm.drf, DRFMetalearner::new),
                new LocalProvider<>(Algorithm.gbm, GBMMetalearner::new),
                new LocalProvider<>(Algorithm.glm, GLMMetalearner::new),
                new LocalProvider<>(Algorithm.naivebayes, NaiveBayesMetalearner::new),
        };
        for (MetalearnerProvider provider : localProviders) {
            providersByName.put(provider.getName(), provider);
        }

        ServiceLoader<MetalearnerProvider> extensionProviders = ServiceLoader.load(MetalearnerProvider.class);
        for (MetalearnerProvider provider : extensionProviders) {
            providersByName.put(provider.getName(), provider);
        }
    }

    static Algorithm getActualMetalearnerAlgo(Algorithm algo) {
        assertAvailable(algo.name());
        return algo == Algorithm.AUTO ? Algorithm.glm : algo;
    }

    static Model.Parameters createParameters(String name) {
        assertAvailable(name);
        return createInstance(name).createBuilder()._parms;
    }

    static Metalearner createInstance(String name) {
        assertAvailable(name);
        return providersByName.get(name).newInstance();
    }

    private static void assertAvailable(String algo) {
        if (!providersByName.containsKey(algo))
            throw new H2OIllegalArgumentException("'"+algo+"' metalearner is not supported or available.");
    }

    /**
     * A local implementation of {@link MetalearnerProvider} to expose the {@link Metalearner}s defined in this class.
     */
    static class LocalProvider<M extends Metalearner> implements MetalearnerProvider<M> {

        private Algorithm _algorithm;
        private Supplier<M> _instanceFactory;

        public LocalProvider(Algorithm algorithm,
                             Supplier<M> instanceFactory) {
            _algorithm = algorithm;
            _instanceFactory = instanceFactory;
        }

        @Override
        public String getName() {
            return _algorithm.name();
        }

        @Override
        public M newInstance() {
            return _instanceFactory.get();
        }
    }

    /**
     * A simple implementation of {@link Metalearner} suitable for any algo; it is just using the algo with its default parameters.
     */
    public static class SimpleMetalearner extends Metalearner {
        private String _algo;

        protected SimpleMetalearner(String algo) {
            _algo = algo;
        }

        @Override
        ModelBuilder createBuilder() {
            return ModelBuilder.make(_algo, _metalearnerJob, _metalearnerKey);
        }

        @Override
        protected void setCustomParams(Model.Parameters parms) {
            super.setCustomParams(parms);
            switch (_parms._distribution) {
                case bernoulli:
                case multinomial:
                case poisson:
                case laplace:
                case gaussian:
                case gamma:
                    setOverridableParm(parms, "distribution", _parms._distribution);
                    break;
                case huber:
                    setOverridableParm(parms, "distribution", _parms._distribution);
                    setOverridableParm(parms, "huber_alpha", _parms._huber_alpha);
                    break;
                case tweedie:
                    setOverridableParm(parms, "distribution", _parms._distribution);
                    setOverridableParm(parms, "tweedie_power", _parms._tweedie_power);
                    break;
                case quantile:
                    setOverridableParm(parms, "distribution", _parms._distribution);
                    setOverridableParm(parms, "quantile_alpha", _parms._quantile_alpha);
                    break;
                case custom:
                    setOverridableParm(parms, "distribution", _parms._distribution);
                    setOverridableParm(parms, "custom_distribution_func", _parms._custom_distribution_func);
                    break;
                default:
                    Log.warn("Metalearner doesn't support distribution \""
                            .concat(_parms._distribution.name())
                            .concat("\""));
            }
        }
    }

    static class DLMetalearner extends SimpleMetalearner {
        public DLMetalearner() {
            super(Algorithm.deeplearning.name());
        }

        @Override
        protected void setCustomParams(Model.Parameters parms) {
            switch (_parms._distribution) {
                case custom:
                case quasibinomial:
                case ordinal:
                case modified_huber:
                    Log.warn("Deep Learning metalearner doesn't support inferred distribution \""
                            .concat(_parms._distribution.name()).concat("\""));
                case AUTO:
                    return;
            }
            super.setCustomParams(parms);
        }
    }

    static class DRFMetalearner extends SimpleMetalearner {
        public DRFMetalearner() {
            super(Algorithm.drf.name());
        }

        @Override
        protected void setCustomParams(Model.Parameters parms) {
            return;
        }
    }

    static class GBMMetalearner extends SimpleMetalearner {
        public GBMMetalearner() {
            super(Algorithm.gbm.name());
        }

        @Override
        protected void setCustomParams(Model.Parameters parms) {
            switch (_parms._distribution) {
                case ordinal:
                case modified_huber:
                    Log.warn("Gradient Boosting Machine metalearner doesn't support inferred distribution \""
                            .concat(_parms._distribution.name()).concat("\""));
                case AUTO:
                    return;
            }
            super.setCustomParams(parms);
        }
    }

    static class GLMMetalearner extends Metalearner<GLM, GLMModel, GLMParameters> {
        @Override
        GLM createBuilder() {
            return ModelBuilder.make("GLM", _metalearnerJob, _metalearnerKey);
        }

        @Override
        protected void setCustomParams(GLMParameters parms) {
            switch (_parms._distribution) {
                case gaussian:
                    setOverridableParm(parms, "family", GLMParameters.Family.gaussian);
                    break;
                case bernoulli:
                    setOverridableParm(parms, "family", GLMParameters.Family.binomial);
                    break;
                case ordinal:
                    setOverridableParm(parms, "family", GLMParameters.Family.ordinal);
                    break;
                case quasibinomial:
                    setOverridableParm(parms, "family", GLMParameters.Family.quasibinomial);
                    break;
                case multinomial:
                    setOverridableParm(parms, "family", GLMParameters.Family.multinomial);
                    break;
                case poisson:
                    setOverridableParm(parms, "family", GLMParameters.Family.poisson);
                    break;
                case gamma:
                    setOverridableParm(parms, "family", GLMParameters.Family.gamma);
                    break;
                case tweedie:
                    setOverridableParm(parms, "family", GLMParameters.Family.tweedie);
                    setOverridableParm(parms, "tweedie_power", _parms._tweedie_power);
                    break;
                default:
                    if (_model.modelCategory == ModelCategory.Regression) {
                        setOverridableParm(parms, "family", GLMParameters.Family.gaussian);
                    } else if (_model.modelCategory == ModelCategory.Binomial) {
                        setOverridableParm(parms, "family", GLMParameters.Family.binomial);
                    } else if (_model.modelCategory == ModelCategory.Multinomial) {
                        setOverridableParm(parms, "family", GLMParameters.Family.multinomial);
                    } else {
                        throw new H2OIllegalArgumentException("Family " + _model.modelCategory + "  is not supported.");
                    }
            }

        }
    }

    static class NaiveBayesMetalearner extends SimpleMetalearner {
        public NaiveBayesMetalearner() {
            super(Algorithm.naivebayes.name());
        }

        @Override
        protected void setCustomParams(Model.Parameters parms) {
            return;
        }
    }

    static class AUTOMetalearner extends GLMMetalearner {

        @Override
        protected void setCustomParams(GLMParameters parms) {
            //add GLM custom params
            super.setCustomParams(parms);

            //specific to AUTO mode
            setOverridableParm(parms, "non_negative", true);
            //parms._alpha = new double[] {0.0, 0.25, 0.5, 0.75, 1.0};

            // feature columns are already homogeneous (probabilities); when standardization is enabled,
            // there can be information loss if some columns have very low probabilities compared with others for example (bad model)
            // giving more weight than it should to those columns.
            setOverridableParm(parms,"standardize",false);

            // Enable lambda search if a validation frame is passed in to get a better GLM fit.
            // Since we are also using non_negative to true, we should also set early_stopping = false.
            if (parms._valid != null) {
                setOverridableParm(parms, "lambda_search", true);
                setOverridableParm(parms, "early_stopping", false);
            }
        }
    }

}
