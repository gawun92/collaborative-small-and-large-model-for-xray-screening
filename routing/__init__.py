from routing.max_score import max_score
from routing.mean_score import mean_score
from routing.entropy import entropy_score
from routing.nms_box_count import nms_box_count_score

ROUTING_METHODS = {
    "max_score":     max_score,
    "mean_score":    mean_score,
    "entropy":       entropy_score,
    "nms_box_count": nms_box_count_score,
}
