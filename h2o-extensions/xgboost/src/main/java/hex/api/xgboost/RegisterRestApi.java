package hex.api.xgboost;

import hex.tree.xgboost.XGBoost;
import hex.tree.xgboost.XGBoostExtension;
import hex.tree.xgboost.remote.RemoteXGBoostHandler;
import water.ExtensionManager;
import water.api.AlgoAbstractRegister;
import water.api.RestApiContext;
import water.api.SchemaServer;

import java.util.Collections;
import java.util.List;

public class RegisterRestApi extends AlgoAbstractRegister {

  @Override
  public void registerEndPoints(RestApiContext context) {
    XGBoostExtension ext = (XGBoostExtension) ExtensionManager.getInstance().getCoreExtension(XGBoostExtension.NAME);
    ext.logNativeLibInfo();
    XGBoost xgBoostMB = new XGBoost(true);
    int version = SchemaServer.getStableVersion();
    // Register XGBoost model builder REST API
    registerModelBuilder(context, xgBoostMB, version);
    // Register Remote XGBoost execution
    context.registerEndpoint(
        "remote_xgb_init", "POST /3/XGBoostExecutor.init",
        RemoteXGBoostHandler.class, "init",
        "Remote XGBoost execution - init"
    );
    context.registerEndpoint(
        "remote_xgb_setup", "POST /3/XGBoostExecutor.setup",
        RemoteXGBoostHandler.class, "setup",
        "Remote XGBoost execution - setup"
    );
    context.registerEndpoint(
        "remote_xgb_update", "POST /3/XGBoostExecutor.update",
        RemoteXGBoostHandler.class, "update",
        "Remote XGBoost execution - update"
    );
    context.registerEndpoint(
        "remote_xgb_features", "POST /3/XGBoostExecutor.getFeatures",
        RemoteXGBoostHandler.class, "getFeatureMap",
        "Remote XGBoost execution - get feature map"
    );
    context.registerEndpoint(
        "remote_xgb_booster", "POST /3/XGBoostExecutor.getBooster",
        RemoteXGBoostHandler.class, "getBooster",
        "Remote XGBoost execution - get booster"
    );
    context.registerEndpoint(
        "remote_xgb_cleanup", "POST /3/XGBoostExecutor.cleanup",
        RemoteXGBoostHandler.class, "cleanup",
        "Remote XGBoost execution - cleanup"
    );
    context.registerEndpoint(
        "remote_xgb_matrix", "POST /3/XGBoostExecutor.getMatrix",
        RemoteXGBoostHandler.class, "getMatrix",
        "Remote XGBoost execution - getMatrix"
    );
  }

  @Override
  public String getName() {
    return "XGBoost";
  }

  @Override
  public List<String> getRequiredCoreExtensions() {
    return Collections.singletonList(XGBoostExtension.NAME);
  }
}
