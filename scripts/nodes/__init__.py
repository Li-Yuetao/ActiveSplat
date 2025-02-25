from activesplat.msg import frame
from activesplat.srv import\
    GetTopdownConfig, GetTopdownConfigResponse, GetTopdownConfigRequest,\
        GetTopdown, GetTopdownResponse, GetTopdownRequest,\
            SetPlannerState, SetPlannerStateResponse, SetPlannerStateRequest,\
                GetDatasetConfig, GetDatasetConfigResponse, GetDatasetConfigRequest,\
                    ResetEnv, ResetEnvResponse, ResetEnvRequest,\
                        SetMapper, SetMapperResponse, SetMapperRequest,\
                            GetOpacity, GetOpacityRequest, GetOpacityResponse,\
                                    GetVoronoiGraph, GetVoronoiGraphRequest, GetVoronoiGraphResponse,\
                                        GetNavPath, GetNavPathRequest, GetNavPathResponse
                            
TURN = 0.2
SPEED = 0.2
USE_RANDOM_SELECTION = False
USE_ROTATION_SELECTION = True
USE_HIGH_CONNECTIVITY = True
USE_HIERARCHICAL_PLAN = True