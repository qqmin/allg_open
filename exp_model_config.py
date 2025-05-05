# import torch
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# SuperPoint 参数
sp_conf = {
    "descriptor_dim": 256,
    "nms_radius": 30,
    "max_num_keypoints": 50,
    "detection_threshold": 0.01,
    "remove_borders": 4,
}

# LightGlue 参数 (针对 SuperPoint 特征的配置)
lg_conf_sp = {
    "name": "lightglue",
    "input_dim": 256,            # 输入特征维度 (通常与权重自动匹配)
    "descriptor_dim": 256,
    "add_scale_ori": False,
    "n_layers": 9,
    "num_heads": 4,
    "flash": True,               # 如果可用则启用 FlashAttention
    "mp": False,                 # 混合精度
    "depth_confidence": 0.99,    # 早停策略 (如不需要可设为 -1)
    "width_confidence": 0.99,    # 点筛选阈值 (如不需要可设为 -1)
    "filter_threshold": 0.001,     # 匹配阈值
    "weights": None,
}

# ALIKED 参数
al_conf = {
    "model_name": "aliked-n16",
    "max_num_keypoints": 10,  # -1  2048      #  无遮挡 21   有遮挡 16
    "detection_threshold": 0.005,  # 0.0002  # 无遮挡 0.012  有遮挡 0.0012
    "nms_radius": 20,
}
# allg 参数
# al_conf = {
#     "model_name": "aliked-n16",
#     "max_num_keypoints": 30,  # -1  2048
#     "detection_threshold": 0.2,  # 0.0002
#     "nms_radius": 20,
# }

# LightGlue 参数 (针对 ALIKED 特征的配置)
lg_conf_al = {
    "name": "lightglue",
    "input_dim": 256,
    "descriptor_dim": 256,
    "add_scale_ori": False,
    "n_layers": 9,
    "num_heads": 4,
    "flash": True,
    "mp": False,
    "depth_confidence": 0.95,
    "width_confidence": 0.99,
    "filter_threshold": 0.001,  # 0.1
    "weights": None,
}


config = {
    "superpoint": sp_conf,
    "lightglue_sp": lg_conf_sp,
    "aliked": al_conf,
    "lightglue_al": lg_conf_al,
}
