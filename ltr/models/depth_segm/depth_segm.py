import torch.nn as nn
import ltr.models.backbone as backbones
import ltr.models.depth_segm as segmmodels
from ltr import model_constructor


class DepthSegmNet(nn.Module):
    """ Segmentation network module"""
    def __init__(self, feature_extractor, segm_predictor, segm_layers, extractor_grad=True, depth_feature_extractor=None):
        """
        args:
            feature_extractor - backbone feature extractor - RGB
            segm_predictor - segmentation module
            segm_layers - List containing the name of the layers from feature_extractor, which are used in segm_predictor
            extractor_grad - Bool indicating whether backbone feature extractor requires gradients
        """
        super(DepthSegmNet, self).__init__()

        self.feature_extractor = feature_extractor
        self.depth_feature_extractor = depth_feature_extractor
        self.segm_predictor = segm_predictor
        self.segm_layers = segm_layers

        if not extractor_grad:
            for p in self.feature_extractor.parameters():
                p.requires_grad_(False)

    def forward(self, train_colors, train_depths, test_colors, test_depths, train_masks,
                    test_dist=None, debug=False, test_raw_d=None, train_raw_d=None):
        """ Forward pass
        Note: If the training is done in sequence mode, that is, test_imgs.dim() == 5, then the batch dimension
        corresponds to the first dimensions. test_imgs is thus of the form [sequence, batch, feature, row, col]
        """

        train_feat_rgb = self.extract_backbone_features(train_colors) # B * C * H * W -> B * C * H * W
        test_feat_rgb = self.extract_backbone_features(test_colors)
        train_feat_rgb = [feat for feat in train_feat_rgb.values()] # layer 0 - 3
        test_feat_rgb = [feat for feat in test_feat_rgb.values()]

        if self.depth_feature_extractor is None:
            train_feat_d = self.segm_predictor.depth_feat_extractor(train_depths)
            test_feat_d = self.segm_predictor.depth_feat_extractor(test_depths)
        else:
            train_feat_d = self.extract_depth_backbone_features(train_depths) # B * C * H * W -> B * C * H * W
            test_feat_d = self.extract_depth_backbone_features(test_depths)
            train_feat_d = [feat for feat in train_feat_d.values()] # layer 0 - 3
            test_feat_d = [feat for feat in test_feat_d.values()]

        train_masks = [train_masks]

        if test_dist is not None:
            test_dist = [test_dist]

        #
        if test_raw_d is not None and train_raw_d is not None:
            segm_pred = self.segm_predictor(test_feat_rgb, test_feat_d,
                                            train_feat_rgb, train_feat_d,
                                            train_masks, test_dist,
                                            test_raw_d=test_raw_d,
                                            train_raw_d=train_raw_d,
                                            debug=debug)
        else:
            segm_pred = self.segm_predictor(test_feat_rgb, test_feat_d,
                                            train_feat_rgb, train_feat_d,
                                            train_masks, test_dist, debug=debug)
        return segm_pred

    def extract_backbone_features(self, im, layers=None):
        if layers is None:
            layers = self.segm_layers
        return self.feature_extractor(im, layers)

    def extract_depth_backbone_features(self, dp, layers=None):
        if layers is None:
            layers = self.segm_layers
        return self.depth_feature_extractor(dp, layers)

    def extract_features(self, im, layers):
        return self.feature_extractor(im, layers)

@model_constructor
def depth_segm_resnet18(segm_input_dim=(256,256), segm_inter_dim=(256,256),
                        backbone_pretrained=True, topk_pos=3, topk_neg=3, mixer_channels=2):
    # backbone
    backbone_net = backbones.resnet18(pretrained=backbone_pretrained)

    # segmentation dimensions
    segm_input_dim = (64, 64, 128, 256)
    segm_inter_dim = (4, 16, 32, 64)
    segm_dim = (64, 64)  # convolutions before cosine similarity

    # segmentation
    segm_predictor = segmmodels.DepthSegmNet()

    net = DepthSegmNet(feature_extractor=backbone_net, segm_predictor=segm_predictor,
                       segm_layers=['conv1', 'layer1', 'layer2', 'layer3'], extractor_grad=False)

    return net


@model_constructor
def depth_segm_resnet50(segm_input_dim=(256,256), segm_inter_dim=(256,256),
                        backbone_pretrained=True, topk_pos=3, topk_neg=3, mixer_channels=2):
    # backbone
    backbone_net = backbones.resnet50(pretrained=backbone_pretrained)

    # segmentation dimensions
    segm_input_dim = (64, 256, 512, 1024)
    segm_inter_dim = (4, 16, 32, 64)
    segm_dim = (64, 64)  # convolutions before cosine similarity

    # segmentation
    segm_predictor = segmmodels.DepthSegmNet()

    net = DepthSegmNet(feature_extractor=backbone_net, segm_predictor=segm_predictor,
                       segm_layers=['conv1', 'layer1', 'layer2', 'layer3'], extractor_grad=False)  # extractor_grad=False

    return net


@model_constructor
def depth_only_segm_resnet50(segm_input_dim=(256,256), segm_inter_dim=(256,256),
                        backbone_pretrained=True, topk_pos=3, topk_neg=3, mixer_channels=2):
    # backbone
    backbone_net = backbones.resnet50(pretrained=backbone_pretrained)

    # segmentation dimensions
    segm_input_dim = (64, 256, 512, 1024)
    segm_inter_dim = (4, 16, 32, 64)
    segm_dim = (64, 64)  # convolutions before cosine similarity

    # segmentation
    segm_predictor = segmmodels.DepthOnlySegmNet()

    net = DepthSegmNet(feature_extractor=backbone_net, segm_predictor=segm_predictor,
                       segm_layers=['conv1', 'layer1', 'layer2', 'layer3'], extractor_grad=False)  # extractor_grad=False

    return net


@model_constructor
def depth_init_segm_resnet50(segm_input_dim=(256,256), segm_inter_dim=(256,256),
                        backbone_pretrained=True, topk_pos=3, topk_neg=3, mixer_channels=2):
    # backbone
    backbone_net = backbones.resnet50(pretrained=backbone_pretrained)

    # segmentation dimensions
    segm_input_dim = (64, 256, 512, 1024)
    segm_inter_dim = (4, 16, 32, 64)
    segm_dim = (64, 64)  # convolutions before cosine similarity

    # segmentation
    segm_predictor = segmmodels.DepthInitSegmNet()

    net = DepthSegmNet(feature_extractor=backbone_net, segm_predictor=segm_predictor,
                       segm_layers=['conv1', 'layer1', 'layer2', 'layer3'], extractor_grad=False)  # extractor_grad=False

    return net


@model_constructor
def depth_segm_attention_resnet50(segm_input_dim=(256,256), segm_inter_dim=(256,256),
                        backbone_pretrained=True, topk_pos=3, topk_neg=3, mixer_channels=2):
    # backbone
    backbone_net = backbones.resnet50(pretrained=backbone_pretrained)

    # segmentation dimensions
    segm_input_dim = (64, 256, 512, 1024)
    segm_inter_dim = (4, 16, 32, 64)
    segm_dim = (64, 64)  # convolutions before cosine similarity

    # segmentation
    segm_predictor = segmmodels.DepthSegmNetAttention()

    net = DepthSegmNet(feature_extractor=backbone_net, segm_predictor=segm_predictor,
                       segm_layers=['conv1', 'layer1', 'layer2', 'layer3'], extractor_grad=False)  # extractor_grad=False

    return net


@model_constructor
def depth_segm_attention01_resnet50(segm_input_dim=(256,256), segm_inter_dim=(256,256),
                        backbone_pretrained=True, topk_pos=3, topk_neg=3, mixer_channels=2):
    # backbone
    backbone_net = backbones.resnet50(pretrained=backbone_pretrained)

    # segmentation dimensions
    segm_input_dim = (64, 256, 512, 1024)
    segm_inter_dim = (4, 16, 32, 64)
    segm_dim = (64, 64)  # convolutions before cosine similarity

    # segmentation
    segm_predictor = segmmodels.DepthSegmNetAttention01()

    net = DepthSegmNet(feature_extractor=backbone_net, segm_predictor=segm_predictor,
                       segm_layers=['conv1', 'layer1', 'layer2', 'layer3'], extractor_grad=False)  # extractor_grad=False

    return net

@model_constructor
def depth_segm_attention02_resnet50(segm_input_dim=(256,256), segm_inter_dim=(256,256),
                        backbone_pretrained=True, topk_pos=3, topk_neg=3, mixer_channels=2):
    # backbone
    backbone_net = backbones.resnet50(pretrained=backbone_pretrained)

    # segmentation dimensions
    segm_input_dim = (64, 256, 512, 1024)
    segm_inter_dim = (4, 16, 32, 64)
    segm_dim = (64, 64)  # convolutions before cosine similarity

    # segmentation
    segm_predictor = segmmodels.DepthSegmNetAttention02()

    net = DepthSegmNet(feature_extractor=backbone_net, segm_predictor=segm_predictor,
                       segm_layers=['conv1', 'layer1', 'layer2', 'layer3'], extractor_grad=False)  # extractor_grad=False

    return net

@model_constructor
def depth_segm_attention03_resnet50(segm_input_dim=(256,256), segm_inter_dim=(256,256),
                        backbone_pretrained=True, topk_pos=3, topk_neg=3, mixer_channels=2):
    # backbone
    backbone_net = backbones.resnet50(pretrained=backbone_pretrained)

    # segmentation dimensions
    segm_input_dim = (64, 256, 512, 1024)
    segm_inter_dim = (4, 16, 32, 64)
    segm_dim = (64, 64)  # convolutions before cosine similarity

    # segmentation
    segm_predictor = segmmodels.DepthSegmNetAttention03()

    net = DepthSegmNet(feature_extractor=backbone_net, segm_predictor=segm_predictor,
                       segm_layers=['conv1', 'layer1', 'layer2', 'layer3'], extractor_grad=False)  # extractor_grad=False

    return net

@model_constructor
def depth_segm_attention04_resnet50(segm_input_dim=(256,256), segm_inter_dim=(256,256),
                        backbone_pretrained=True, topk_pos=3, topk_neg=3, mixer_channels=2):
    # backbone
    backbone_net = backbones.resnet50(pretrained=backbone_pretrained)

    # segmentation dimensions
    segm_input_dim = (64, 256, 512, 1024)
    segm_inter_dim = (4, 16, 32, 64)
    segm_dim = (64, 64)  # convolutions before cosine similarity

    # segmentation
    segm_predictor = segmmodels.DepthSegmNetAttention04()

    net = DepthSegmNet(feature_extractor=backbone_net, segm_predictor=segm_predictor,
                       segm_layers=['conv1', 'layer1', 'layer2', 'layer3'], extractor_grad=False)  # extractor_grad=False

    return net


@model_constructor
def depth_segm_attention05_resnet50(segm_input_dim=(256,256), segm_inter_dim=(256,256),
                        backbone_pretrained=True, topk_pos=3, topk_neg=3, mixer_channels=2):
    # backbone
    backbone_net = backbones.resnet50(pretrained=backbone_pretrained)

    # segmentation dimensions
    segm_input_dim = (64, 256, 512, 1024)
    segm_inter_dim = (4, 16, 32, 64)
    segm_dim = (64, 64)  # convolutions before cosine similarity

    # segmentation
    segm_predictor = segmmodels.DepthSegmNetAttention05()

    net = DepthSegmNet(feature_extractor=backbone_net, segm_predictor=segm_predictor,
                       segm_layers=['conv1', 'layer1', 'layer2', 'layer3'], extractor_grad=False)  # extractor_grad=False

    return net

@model_constructor
def depth_segm_attention06_resnet50(segm_input_dim=(256,256), segm_inter_dim=(256,256),
                        backbone_pretrained=True, topk_pos=3, topk_neg=3, mixer_channels=2):
    # backbone
    backbone_net = backbones.resnet50(pretrained=backbone_pretrained)

    # segmentation dimensions
    segm_input_dim = (64, 256, 512, 1024)
    segm_inter_dim = (4, 16, 32, 64)
    segm_dim = (64, 64)  # convolutions before cosine similarity

    # segmentation
    segm_predictor = segmmodels.DepthSegmNetAttention06()

    net = DepthSegmNet(feature_extractor=backbone_net, segm_predictor=segm_predictor,
                       segm_layers=['conv1', 'layer1', 'layer2', 'layer3'], extractor_grad=False)  # extractor_grad=False

    return net

@model_constructor
def depth_segm_attention07_resnet50(segm_input_dim=(256,256), segm_inter_dim=(256,256),
                        backbone_pretrained=True, topk_pos=3, topk_neg=3, mixer_channels=2):
    # backbone
    backbone_net = backbones.resnet50(pretrained=backbone_pretrained)

    # segmentation dimensions
    segm_input_dim = (64, 256, 512, 1024)
    segm_inter_dim = (4, 16, 32, 64)
    segm_dim = (64, 64)  # convolutions before cosine similarity

    # segmentation
    segm_predictor = segmmodels.DepthSegmNetAttention07()

    net = DepthSegmNet(feature_extractor=backbone_net, segm_predictor=segm_predictor,
                       segm_layers=['conv1', 'layer1', 'layer2', 'layer3'], extractor_grad=False)  # extractor_grad=False

    return net

@model_constructor
def depth_segm_attention08_resnet50(segm_input_dim=(256,256), segm_inter_dim=(256,256),
                        backbone_pretrained=True, topk_pos=3, topk_neg=3, mixer_channels=2):
    # backbone
    backbone_net = backbones.resnet50(pretrained=backbone_pretrained)

    # segmentation dimensions
    segm_input_dim = (64, 256, 512, 1024)
    segm_inter_dim = (4, 16, 32, 64)
    segm_dim = (64, 64)  # convolutions before cosine similarity

    # segmentation
    segm_predictor = segmmodels.DepthSegmNetAttention08()

    net = DepthSegmNet(feature_extractor=backbone_net, segm_predictor=segm_predictor,
                       segm_layers=['conv1', 'layer1', 'layer2', 'layer3'], extractor_grad=False)  # extractor_grad=False

    return net


@model_constructor
def depth_segm_attention09_resnet50(segm_input_dim=(256,256), segm_inter_dim=(256,256),
                        backbone_pretrained=True, topk_pos=3, topk_neg=3, mixer_channels=2):
    # backbone
    backbone_net = backbones.resnet50(pretrained=backbone_pretrained)

    # segmentation dimensions
    segm_input_dim = (64, 256, 512, 1024)
    segm_inter_dim = (4, 16, 32, 64)
    segm_dim = (64, 64)  # convolutions before cosine similarity

    # segmentation
    segm_predictor = segmmodels.DepthSegmNetAttention09()

    net = DepthSegmNet(feature_extractor=backbone_net, segm_predictor=segm_predictor,
                       segm_layers=['conv1', 'layer1', 'layer2', 'layer3'], extractor_grad=False)  # extractor_grad=False

    return net

@model_constructor
def depth_segm_attention10_resnet50(segm_input_dim=(256,256), segm_inter_dim=(256,256),
                        backbone_pretrained=True, topk_pos=3, topk_neg=3, mixer_channels=2):
    # backbone
    backbone_net = backbones.resnet50(pretrained=backbone_pretrained)

    # segmentation dimensions
    segm_input_dim = (64, 256, 512, 1024)
    segm_inter_dim = (4, 16, 32, 64)
    segm_dim = (64, 64)  # convolutions before cosine similarity

    # segmentation
    segm_predictor = segmmodels.DepthSegmNetAttention10()

    net = DepthSegmNet(feature_extractor=backbone_net, segm_predictor=segm_predictor,
                       segm_layers=['conv1', 'layer1', 'layer2', 'layer3'], extractor_grad=False)  # extractor_grad=False

    return net

@model_constructor
def depth_segm_attention11_resnet50(segm_input_dim=(256,256), segm_inter_dim=(256,256),
                        backbone_pretrained=True, topk_pos=3, topk_neg=3, mixer_channels=2):
    # backbone
    backbone_net = backbones.resnet50(pretrained=backbone_pretrained)

    # segmentation dimensions
    segm_input_dim = (64, 256, 512, 1024)
    segm_inter_dim = (4, 16, 32, 64)
    segm_dim = (64, 64)  # convolutions before cosine similarity

    # segmentation
    segm_predictor = segmmodels.DepthSegmNetAttention11()

    net = DepthSegmNet(feature_extractor=backbone_net, segm_predictor=segm_predictor,
                       segm_layers=['conv1', 'layer1', 'layer2', 'layer3'], extractor_grad=False)  # extractor_grad=False

    return net



@model_constructor
def depth_segm_attention12_resnet50(segm_input_dim=(256,256), segm_inter_dim=(256,256),
                        backbone_pretrained=True, topk_pos=3, topk_neg=3, mixer_channels=2):
    # backbone
    backbone_net = backbones.resnet50(pretrained=backbone_pretrained)

    # segmentation dimensions
    segm_input_dim = (64, 256, 512, 1024)
    segm_inter_dim = (4, 16, 32, 64)
    segm_dim = (64, 64)  # convolutions before cosine similarity

    # segmentation
    segm_predictor = segmmodels.DepthSegmNetAttention12()

    net = DepthSegmNet(feature_extractor=backbone_net, segm_predictor=segm_predictor,
                       segm_layers=['conv1', 'layer1', 'layer2', 'layer3'], extractor_grad=False)  # extractor_grad=False

    return net

@model_constructor
def depth_segm_attention13_resnet50(segm_input_dim=(256,256), segm_inter_dim=(256,256),
                        backbone_pretrained=True, topk_pos=3, topk_neg=3, mixer_channels=2):
    # backbone
    backbone_net = backbones.resnet50(pretrained=backbone_pretrained)

    # segmentation dimensions
    segm_input_dim = (64, 256, 512, 1024)
    segm_inter_dim = (4, 16, 32, 64)
    segm_dim = (64, 64)  # convolutions before cosine similarity

    # segmentation
    segm_predictor = segmmodels.DepthSegmNetAttention13()

    net = DepthSegmNet(feature_extractor=backbone_net, segm_predictor=segm_predictor,
                       segm_layers=['conv1', 'layer1', 'layer2', 'layer3'], extractor_grad=False)  # extractor_grad=False

    return net



@model_constructor
def depth_segm_attention02_1_resnet50(segm_input_dim=(256,256), segm_inter_dim=(256,256),
                        backbone_pretrained=True, topk_pos=3, topk_neg=3, mixer_channels=2):
    # backbone
    backbone_net = backbones.resnet50(pretrained=backbone_pretrained)

    # segmentation dimensions
    segm_input_dim = (64, 256, 512, 1024)
    segm_inter_dim = (4, 16, 32, 64)
    segm_dim = (64, 64)  # convolutions before cosine similarity

    # segmentation
    segm_predictor = segmmodels.DepthSegmNetAttention02_1()

    net = DepthSegmNet(feature_extractor=backbone_net, segm_predictor=segm_predictor,
                       segm_layers=['conv1', 'layer1', 'layer2', 'layer3'], extractor_grad=False)  # extractor_grad=False

    return net



@model_constructor
def depth_segm_attention02_2_resnet50(segm_input_dim=(256,256), segm_inter_dim=(256,256),
                        backbone_pretrained=True, topk_pos=3, topk_neg=3, mixer_channels=2):
    # backbone
    backbone_net = backbones.resnet50(pretrained=backbone_pretrained)

    # segmentation dimensions
    segm_input_dim = (64, 256, 512, 1024)
    segm_inter_dim = (4, 16, 32, 64)
    segm_dim = (64, 64)  # convolutions before cosine similarity

    # segmentation
    segm_predictor = segmmodels.DepthSegmNetAttention02_2()

    net = DepthSegmNet(feature_extractor=backbone_net, segm_predictor=segm_predictor,
                       segm_layers=['conv1', 'layer1', 'layer2', 'layer3'], extractor_grad=False)  # extractor_grad=False

    return net

@model_constructor
def depth_segm_attention02_3_resnet50(segm_input_dim=(256,256), segm_inter_dim=(256,256),
                        backbone_pretrained=True, topk_pos=3, topk_neg=3, mixer_channels=2):
    # backbone
    backbone_net = backbones.resnet50(pretrained=backbone_pretrained)

    # segmentation dimensions
    segm_input_dim = (64, 256, 512, 1024)
    segm_inter_dim = (4, 16, 32, 64)
    segm_dim = (64, 64)  # convolutions before cosine similarity

    # segmentation
    segm_predictor = segmmodels.DepthSegmNetAttention02_3()

    net = DepthSegmNet(feature_extractor=backbone_net, segm_predictor=segm_predictor,
                       segm_layers=['conv1', 'layer1', 'layer2', 'layer3'], extractor_grad=False)  # extractor_grad=False

    return net

@model_constructor
def depth_segm_attention02_4_resnet50(segm_input_dim=(256,256), segm_inter_dim=(256,256),
                        backbone_pretrained=True, topk_pos=3, topk_neg=3, mixer_channels=2):
    # backbone
    backbone_net = backbones.resnet50(pretrained=backbone_pretrained)

    # segmentation dimensions
    segm_input_dim = (64, 256, 512, 1024)
    segm_inter_dim = (4, 16, 32, 64)
    segm_dim = (64, 64)  # convolutions before cosine similarity

    # segmentation
    segm_predictor = segmmodels.DepthSegmNetAttention02_4()

    net = DepthSegmNet(feature_extractor=backbone_net, segm_predictor=segm_predictor,
                       segm_layers=['conv1', 'layer1', 'layer2', 'layer3'], extractor_grad=False)  # extractor_grad=False

    return net

@model_constructor
def depth_segm_attention02_5_resnet50(segm_input_dim=(256,256), segm_inter_dim=(256,256),
                        backbone_pretrained=True, topk_pos=3, topk_neg=3, mixer_channels=2):
    # backbone
    backbone_net = backbones.resnet50(pretrained=backbone_pretrained)

    # segmentation dimensions
    segm_input_dim = (64, 256, 512, 1024)
    segm_inter_dim = (4, 16, 32, 64)
    segm_dim = (64, 64)  # convolutions before cosine similarity

    # segmentation
    segm_predictor = segmmodels.DepthSegmNetAttention02_5()

    net = DepthSegmNet(feature_extractor=backbone_net, segm_predictor=segm_predictor,
                       segm_layers=['conv1', 'layer1', 'layer2', 'layer3'], extractor_grad=False)  # extractor_grad=False

    return net

@model_constructor
def depth_segm_attention02_6_resnet50(segm_input_dim=(256,256), segm_inter_dim=(256,256),
                        backbone_pretrained=True, topk_pos=3, topk_neg=3, mixer_channels=2):
    # backbone
    backbone_net = backbones.resnet50(pretrained=backbone_pretrained)

    # segmentation dimensions
    segm_input_dim = (64, 256, 512, 1024)
    segm_inter_dim = (4, 16, 32, 64)
    segm_dim = (64, 64)  # convolutions before cosine similarity

    # segmentation
    segm_predictor = segmmodels.DepthSegmNetAttention02_6()

    net = DepthSegmNet(feature_extractor=backbone_net, segm_predictor=segm_predictor,
                       segm_layers=['conv1', 'layer1', 'layer2', 'layer3'], extractor_grad=False)  # extractor_grad=False

    return net

@model_constructor
def depth_segm_attention02_7_resnet50(segm_input_dim=(256,256), segm_inter_dim=(256,256),
                        backbone_pretrained=True, topk_pos=3, topk_neg=3, mixer_channels=2):
    # backbone
    backbone_net = backbones.resnet50(pretrained=backbone_pretrained)

    # segmentation dimensions
    segm_input_dim = (64, 256, 512, 1024)
    segm_inter_dim = (4, 16, 32, 64)
    segm_dim = (64, 64)  # convolutions before cosine similarity

    # segmentation
    segm_predictor = segmmodels.DepthSegmNetAttention02_7()

    net = DepthSegmNet(feature_extractor=backbone_net, segm_predictor=segm_predictor,
                       segm_layers=['conv1', 'layer1', 'layer2', 'layer3'], extractor_grad=False)  # extractor_grad=False

    return net

@model_constructor
def depth_segm_attention02_8_resnet50(segm_input_dim=(256,256), segm_inter_dim=(256,256),
                        backbone_pretrained=True, topk_pos=3, topk_neg=3, mixer_channels=2):
    # backbone
    backbone_net = backbones.resnet50(pretrained=backbone_pretrained)

    # segmentation dimensions
    segm_input_dim = (64, 256, 512, 1024)
    segm_inter_dim = (4, 16, 32, 64)
    segm_dim = (64, 64)  # convolutions before cosine similarity

    # segmentation
    segm_predictor = segmmodels.DepthSegmNetAttention02_8()

    net = DepthSegmNet(feature_extractor=backbone_net, segm_predictor=segm_predictor,
                       segm_layers=['conv1', 'layer1', 'layer2', 'layer3'], extractor_grad=False)  # extractor_grad=False

    return net


@model_constructor
def depth_segm_attention02_8DC_resnet50(segm_input_dim=(256,256), segm_inter_dim=(256,256),
                        backbone_pretrained=True, topk_pos=3, topk_neg=3, mixer_channels=2):
    # backbone
    backbone_net = backbones.resnet50(pretrained=backbone_pretrained)

    # segmentation dimensions
    segm_input_dim = (64, 256, 512, 1024)
    segm_inter_dim = (4, 16, 32, 64)
    segm_dim = (64, 64)  # convolutions before cosine similarity

    # segmentation
    segm_predictor = segmmodels.DepthSegmNetAttention02_8DC()

    net = DepthSegmNet(feature_extractor=backbone_net, segm_predictor=segm_predictor,
                       segm_layers=['conv1', 'layer1', 'layer2', 'layer3'], extractor_grad=False)  # extractor_grad=False

    return net

@model_constructor
def depth_segm_attention02_8DC_Max_resnet50(segm_input_dim=(256,256), segm_inter_dim=(256,256),
                        backbone_pretrained=True, topk_pos=3, topk_neg=3, mixer_channels=2):
    # backbone
    backbone_net = backbones.resnet50(pretrained=backbone_pretrained)

    # segmentation dimensions
    # segm_input_dim = (64, 256, 512, 1024)
    # segm_inter_dim = (4, 16, 32, 64)
    # segm_dim = (64, 64)  # convolutions before cosine similarity

    # segmentation
    segm_predictor = segmmodels.DepthSegmNetAttention02_8DC_Max()

    net = DepthSegmNet(feature_extractor=backbone_net, segm_predictor=segm_predictor,
                       segm_layers=['conv1', 'layer1', 'layer2', 'layer3'], extractor_grad=False)  # extractor_grad=False

    return net


@model_constructor
def depth_segm_attention02_8DC_Sum_resnet50(segm_input_dim=(256,256), segm_inter_dim=(256,256),
                        backbone_pretrained=True, topk_pos=3, topk_neg=3, mixer_channels=2):
    # backbone
    backbone_net = backbones.resnet50(pretrained=backbone_pretrained)

    # segmentation dimensions
    # segm_input_dim = (64, 256, 512, 1024)
    # segm_inter_dim = (4, 16, 32, 64)
    # segm_dim = (64, 64)  # convolutions before cosine similarity

    # segmentation
    segm_predictor = segmmodels.DepthSegmNetAttention02_8DC_Sum()

    net = DepthSegmNet(feature_extractor=backbone_net, segm_predictor=segm_predictor,
                       segm_layers=['conv1', 'layer1', 'layer2', 'layer3'], extractor_grad=False)  # extractor_grad=False

    return net

@model_constructor
def depth_segm_attention02_9_resnet50(segm_input_dim=(256,256), segm_inter_dim=(256,256),
                        backbone_pretrained=True, topk_pos=3, topk_neg=3, mixer_channels=2):
    # backbone
    backbone_net = backbones.resnet50(pretrained=backbone_pretrained)

    # segmentation dimensions
    segm_input_dim = (64, 256, 512, 1024)
    segm_inter_dim = (4, 16, 32, 64)
    segm_dim = (64, 64)  # convolutions before cosine similarity

    # segmentation
    segm_predictor = segmmodels.DepthSegmNetAttention02_9()

    net = DepthSegmNet(feature_extractor=backbone_net, segm_predictor=segm_predictor,
                       segm_layers=['conv1', 'layer1', 'layer2', 'layer3'], extractor_grad=False)  # extractor_grad=False

    return net


@model_constructor
def depth_segm_RDF_resnet50(segm_input_dim=(256,256), segm_inter_dim=(256,256),
                        backbone_pretrained=True, topk_pos=3, topk_neg=3, mixer_channels=2):
    # backbone
    backbone_net = backbones.resnet50(pretrained=backbone_pretrained)

    # segmentation dimensions
    segm_input_dim = (64, 256, 512, 1024)
    segm_inter_dim = (4, 16, 32, 64)
    segm_dim = (64, 64)  # convolutions before cosine similarity

    # segmentation
    segm_predictor = segmmodels.DepthSegmNet_RDF()

    net = DepthSegmNet(feature_extractor=backbone_net, segm_predictor=segm_predictor,
                       segm_layers=['conv1', 'layer1', 'layer2', 'layer3'], extractor_grad=False)  # extractor_grad=False

    return net

@model_constructor
def depth_segm_attention02_8MMF_resnet50(segm_input_dim=(256,256), segm_inter_dim=(256,256),
                        backbone_pretrained=True, topk_pos=3, topk_neg=3, mixer_channels=2):
    # backbone
    backbone_net = backbones.resnet50(pretrained=backbone_pretrained)

    # segmentation dimensions
    segm_input_dim = (64, 256, 512, 1024)
    segm_inter_dim = (4, 16, 32, 64)
    segm_dim = (64, 64)  # convolutions before cosine similarity

    # segmentation
    segm_predictor = segmmodels.DepthSegmNetAttention02_8MMF()

    net = DepthSegmNet(feature_extractor=backbone_net, segm_predictor=segm_predictor,
                       segm_layers=['conv1', 'layer1', 'layer2', 'layer3'], extractor_grad=False)  # extractor_grad=False

    return net


@model_constructor
def depth_segm_attention02_8MMF_MP_resnet50(segm_input_dim=(256,256), segm_inter_dim=(256,256),
                        backbone_pretrained=True, topk_pos=3, topk_neg=3, mixer_channels=2):
    # backbone
    backbone_net = backbones.resnet50(pretrained=backbone_pretrained)

    # segmentation dimensions
    segm_input_dim = (64, 256, 512, 1024)
    segm_inter_dim = (4, 16, 32, 64)
    segm_dim = (64, 64)  # convolutions before cosine similarity

    # segmentation
    segm_predictor = segmmodels.DepthSegmNetAttention02_8MMF_MP()

    net = DepthSegmNet(feature_extractor=backbone_net, segm_predictor=segm_predictor,
                       segm_layers=['conv1', 'layer1', 'layer2', 'layer3'], extractor_grad=False)  # extractor_grad=False

    return net

@model_constructor
def depth_segm_attention02_8ACNet_MP_resnet50(segm_input_dim=(256,256), segm_inter_dim=(256,256),
                        backbone_pretrained=True, topk_pos=3, topk_neg=3, mixer_channels=2):
    # backbone
    backbone_net = backbones.resnet50(pretrained=backbone_pretrained)

    # segmentation dimensions
    segm_input_dim = (64, 256, 512, 1024)
    segm_inter_dim = (4, 16, 32, 64)
    segm_dim = (64, 64)  # convolutions before cosine similarity

    # segmentation
    segm_predictor = segmmodels.DepthSegmNetAttention02_8ACNet_MP()

    net = DepthSegmNet(feature_extractor=backbone_net, segm_predictor=segm_predictor,
                       segm_layers=['conv1', 'layer1', 'layer2', 'layer3'], extractor_grad=False)  # extractor_grad=False

    return net

@model_constructor
def depth_segm_attention02_8ACNet_MP_woBG_resnet50(segm_input_dim=(256,256), segm_inter_dim=(256,256),
                        backbone_pretrained=True, topk_pos=3, topk_neg=3, mixer_channels=2):
    # backbone
    backbone_net = backbones.resnet50(pretrained=backbone_pretrained)

    # segmentation dimensions
    segm_input_dim = (64, 256, 512, 1024)
    segm_inter_dim = (4, 16, 32, 64)
    segm_dim = (64, 64)  # convolutions before cosine similarity

    # segmentation
    segm_predictor = segmmodels.DepthSegmNetAttention02_8ACNet_MP_woBG()

    net = DepthSegmNet(feature_extractor=backbone_net, segm_predictor=segm_predictor,
                       segm_layers=['conv1', 'layer1', 'layer2', 'layer3'], extractor_grad=False)  # extractor_grad=False

    return net


@model_constructor
def depth_segm_D3S_ACNet_MP_resnet50(segm_input_dim=(256,256), segm_inter_dim=(256,256),
                        backbone_pretrained=True, topk_pos=3, topk_neg=3, mixer_channels=2):
    # backbone
    backbone_net = backbones.resnet50(pretrained=backbone_pretrained)

    # segmentation dimensions
    segm_input_dim = (64, 256, 512, 1024)
    segm_inter_dim = (4, 16, 32, 64)
    segm_dim = (64, 64)  # convolutions before cosine similarity

    # segmentation
    segm_predictor = segmmodels.D3S_RGBD()

    net = DepthSegmNet(feature_extractor=backbone_net, segm_predictor=segm_predictor,
                       segm_layers=['conv1', 'layer1', 'layer2', 'layer3'], extractor_grad=False)  # extractor_grad=False

    return net

@model_constructor
def depth_segm_D3S_CBAM_MP_resnet50(segm_input_dim=(256,256), segm_inter_dim=(256,256),
                        backbone_pretrained=True, topk_pos=3, topk_neg=3, mixer_channels=2):
    # backbone
    backbone_net = backbones.resnet50(pretrained=backbone_pretrained)

    # segmentation dimensions
    segm_input_dim = (64, 256, 512, 1024)
    segm_inter_dim = (4, 16, 32, 64)
    segm_dim = (64, 64)  # convolutions before cosine similarity

    # segmentation
    segm_predictor = segmmodels.D3S_CBAM_RGBD()

    net = DepthSegmNet(feature_extractor=backbone_net, segm_predictor=segm_predictor,
                       segm_layers=['conv1', 'layer1', 'layer2', 'layer3'], extractor_grad=False)  # extractor_grad=False

    return net


@model_constructor
def depth_segm_D3S_CBAM02_MP_resnet50(segm_input_dim=(256,256), segm_inter_dim=(256,256),
                        backbone_pretrained=True, topk_pos=3, topk_neg=3, mixer_channels=2):
    # backbone
    backbone_net = backbones.resnet50(pretrained=backbone_pretrained)

    # segmentation dimensions
    segm_input_dim = (64, 256, 512, 1024)
    segm_inter_dim = (4, 16, 32, 64)
    segm_dim = (64, 64)  # convolutions before cosine similarity

    # segmentation
    segm_predictor = segmmodels.D3S_CBAM_RGBD02()

    net = DepthSegmNet(feature_extractor=backbone_net, segm_predictor=segm_predictor,
                       segm_layers=['conv1', 'layer1', 'layer2', 'layer3'], extractor_grad=False)  # extractor_grad=False

    return net

@model_constructor
def depth_segm_D3S_CBAM02_2_MP_resnet50(segm_input_dim=(256,256), segm_inter_dim=(256,256),
                        backbone_pretrained=True, topk_pos=3, topk_neg=3, mixer_channels=2):
    # backbone
    backbone_net = backbones.resnet50(pretrained=backbone_pretrained)

    # segmentation dimensions
    segm_input_dim = (64, 256, 512, 1024)
    segm_inter_dim = (4, 16, 32, 64)
    segm_dim = (64, 64)  # convolutions before cosine similarity

    # segmentation
    segm_predictor = segmmodels.D3S_CBAM_RGBD02_2()

    net = DepthSegmNet(feature_extractor=backbone_net, segm_predictor=segm_predictor,
                       segm_layers=['conv1', 'layer1', 'layer2', 'layer3'], extractor_grad=False)  # extractor_grad=False

    return net

@model_constructor
def depth_segm_D3S_CBAM03_MP_resnet50(segm_input_dim=(256,256), segm_inter_dim=(256,256),
                        backbone_pretrained=True, topk_pos=3, topk_neg=3, mixer_channels=2):
    # backbone
    backbone_net = backbones.resnet50(pretrained=backbone_pretrained)

    # segmentation dimensions
    segm_input_dim = (64, 256, 512, 1024)
    segm_inter_dim = (4, 16, 32, 64)
    segm_dim = (64, 64)  # convolutions before cosine similarity

    # segmentation
    segm_predictor = segmmodels.D3S_CBAM_RGBD03()

    net = DepthSegmNet(feature_extractor=backbone_net, segm_predictor=segm_predictor,
                       segm_layers=['conv1', 'layer1', 'layer2', 'layer3'], extractor_grad=False)  # extractor_grad=False

    return net

@model_constructor
def depth_segm_D3S_Attention_MP_resnet50(segm_input_dim=(256,256), segm_inter_dim=(256,256),
                        backbone_pretrained=True, topk_pos=3, topk_neg=3, mixer_channels=2):
    # backbone
    backbone_net = backbones.resnet50(pretrained=backbone_pretrained)

    # segmentation dimensions
    segm_input_dim = (64, 256, 512, 1024)
    segm_inter_dim = (4, 16, 32, 64)
    segm_dim = (64, 64)  # convolutions before cosine similarity

    # segmentation
    segm_predictor = segmmodels.D3S_Attention_RGBD()

    net = DepthSegmNet(feature_extractor=backbone_net, segm_predictor=segm_predictor,
                       segm_layers=['conv1', 'layer1', 'layer2', 'layer3'], extractor_grad=False)  # extractor_grad=False

    return net



@model_constructor
def depth_segm_D3S_DW_MP_resnet50(segm_input_dim=(256,256), segm_inter_dim=(256,256),
                        backbone_pretrained=True, topk_pos=3, topk_neg=3, mixer_channels=2):
    # backbone
    backbone_net = backbones.resnet50(pretrained=backbone_pretrained)

    # segmentation dimensions
    segm_input_dim = (64, 256, 512, 1024)
    segm_inter_dim = (4, 16, 32, 64)
    segm_dim = (64, 64)  # convolutions before cosine similarity

    # segmentation
    segm_predictor = segmmodels.D3S_DW_RGBD()

    net = DepthSegmNet(feature_extractor=backbone_net, segm_predictor=segm_predictor,
                       segm_layers=['conv1', 'layer1', 'layer2', 'layer3'], extractor_grad=False)  # extractor_grad=False

    return net


@model_constructor
def depth_segm_D3S_DW02_MP_resnet50(segm_input_dim=(256,256), segm_inter_dim=(256,256),
                        backbone_pretrained=True, topk_pos=3, topk_neg=3, mixer_channels=2):
    # backbone
    backbone_net = backbones.resnet50(pretrained=backbone_pretrained)

    # segmentation dimensions
    segm_input_dim = (64, 256, 512, 1024)
    segm_inter_dim = (4, 16, 32, 64)
    segm_dim = (64, 64)  # convolutions before cosine similarity

    # segmentation
    segm_predictor = segmmodels.D3S_DW02_RGBD()

    net = DepthSegmNet(feature_extractor=backbone_net, segm_predictor=segm_predictor,
                       segm_layers=['conv1', 'layer1', 'layer2', 'layer3'], extractor_grad=False)  # extractor_grad=False

    return net

@model_constructor
def depth_segm_D3S_DW03_MP_resnet50(segm_input_dim=(256,256), segm_inter_dim=(256,256),
                        backbone_pretrained=True, topk_pos=3, topk_neg=3, mixer_channels=2):
    # backbone
    backbone_net = backbones.resnet50(pretrained=backbone_pretrained)

    # segmentation dimensions
    segm_input_dim = (64, 256, 512, 1024)
    segm_inter_dim = (4, 16, 32, 64)
    segm_dim = (64, 64)  # convolutions before cosine similarity

    # segmentation
    segm_predictor = segmmodels.D3S_DW03_RGBD()

    net = DepthSegmNet(feature_extractor=backbone_net, segm_predictor=segm_predictor,
                       segm_layers=['conv1', 'layer1', 'layer2', 'layer3'], extractor_grad=False)  # extractor_grad=False

    return net



@model_constructor
def depth_segm_D3S_DW03_RGBD_resnet50(segm_input_dim=(256,256), segm_inter_dim=(256,256),
                        backbone_pretrained=True, topk_pos=3, topk_neg=3, mixer_channels=2):
    # backbone
    backbone_net = backbones.resnet50(pretrained=backbone_pretrained)
    depth_backbone_net = backbones.resnet50(pretrained=backbone_pretrained,
                                            net_path='/home/sgn/Data1/yan/d3s/checkpoints/ltr/DeT/DeT_DiMP50_Max.pth')

    # segmentation dimensions
    segm_input_dim = (64, 256, 512, 1024)
    segm_inter_dim = (4, 16, 32, 64)
    segm_dim = (64, 64)  # convolutions before cosine similarity

    # segmentation
    segm_predictor = segmmodels.D3S_DW03_RGBD_Resnet_feat()

    net = DepthSegmNet(feature_extractor=backbone_net, segm_predictor=segm_predictor,
                       segm_layers=['conv1', 'layer1', 'layer2', 'layer3'], extractor_grad=False,
                       depth_feature_extractor=depth_backbone_net)

    return net


@model_constructor
def depth_segm_D3S_DW03_1_MP_resnet50(segm_input_dim=(256,256), segm_inter_dim=(256,256),
                        backbone_pretrained=True, topk_pos=3, topk_neg=3, mixer_channels=2):
    # backbone
    backbone_net = backbones.resnet50(pretrained=backbone_pretrained)

    # segmentation dimensions
    segm_input_dim = (64, 256, 512, 1024)
    segm_inter_dim = (4, 16, 32, 64)
    segm_dim = (64, 64)  # convolutions before cosine similarity

    # segmentation
    segm_predictor = segmmodels.D3S_DW03_1_RGBD()

    net = DepthSegmNet(feature_extractor=backbone_net, segm_predictor=segm_predictor,
                       segm_layers=['conv1', 'layer1', 'layer2', 'layer3'], extractor_grad=False)  # extractor_grad=False

    return net


@model_constructor
def depth_segm_D3S_DW04_MP_resnet50(segm_input_dim=(256,256), segm_inter_dim=(256,256),
                        backbone_pretrained=True, topk_pos=3, topk_neg=3, mixer_channels=2):
    # backbone
    backbone_net = backbones.resnet50(pretrained=backbone_pretrained)

    # segmentation dimensions
    segm_input_dim = (64, 256, 512, 1024)
    segm_inter_dim = (4, 16, 32, 64)
    segm_dim = (64, 64)  # convolutions before cosine similarity

    # segmentation
    segm_predictor = segmmodels.D3S_DW04_RGBD()

    net = DepthSegmNet(feature_extractor=backbone_net, segm_predictor=segm_predictor,
                       segm_layers=['conv1', 'layer1', 'layer2', 'layer3'], extractor_grad=False)  # extractor_grad=False

    return net


@model_constructor
def depth_segm_D3S_DW05_MP_resnet50(segm_input_dim=(256,256), segm_inter_dim=(256,256),
                        backbone_pretrained=True, topk_pos=3, topk_neg=3, mixer_channels=2):
    # backbone
    backbone_net = backbones.resnet50(pretrained=backbone_pretrained)

    # segmentation dimensions
    segm_input_dim = (64, 256, 512, 1024)
    segm_inter_dim = (4, 16, 32, 64)
    segm_dim = (64, 64)  # convolutions before cosine similarity

    # segmentation
    segm_predictor = segmmodels.D3S_DW05_RGBD()

    net = DepthSegmNet(feature_extractor=backbone_net, segm_predictor=segm_predictor,
                       segm_layers=['conv1', 'layer1', 'layer2', 'layer3'], extractor_grad=False)  # extractor_grad=False

    return net

@model_constructor
def depth_segm_D3S_DW06_MP_resnet50(segm_input_dim=(256,256), segm_inter_dim=(256,256),
                        backbone_pretrained=True, topk_pos=3, topk_neg=3, mixer_channels=2):
    # backbone
    backbone_net = backbones.resnet50(pretrained=backbone_pretrained)

    # segmentation dimensions
    segm_input_dim = (64, 256, 512, 1024)
    segm_inter_dim = (4, 16, 32, 64)
    segm_dim = (64, 64)  # convolutions before cosine similarity

    # segmentation
    segm_predictor = segmmodels.D3S_DW06_RGBD()

    net = DepthSegmNet(feature_extractor=backbone_net, segm_predictor=segm_predictor,
                       segm_layers=['conv1', 'layer1', 'layer2', 'layer3'], extractor_grad=False)  # extractor_grad=False

    return net

@model_constructor
def depth_segm_D3S_Attn_DW_MP_resnet50(segm_input_dim=(256,256), segm_inter_dim=(256,256),
                        backbone_pretrained=True, topk_pos=3, topk_neg=3, mixer_channels=2):
    # backbone
    backbone_net = backbones.resnet50(pretrained=backbone_pretrained)

    # segmentation dimensions
    segm_input_dim = (64, 256, 512, 1024)
    segm_inter_dim = (4, 16, 32, 64)
    segm_dim = (64, 64)  # convolutions before cosine similarity

    # segmentation
    segm_predictor = segmmodels.D3S_Attn_DW_RGBD()

    net = DepthSegmNet(feature_extractor=backbone_net, segm_predictor=segm_predictor,
                       segm_layers=['conv1', 'layer1', 'layer2', 'layer3'], extractor_grad=False)  # extractor_grad=False

    return net

@model_constructor
def depth_segm_D3S_Attn02_DW_MP_resnet50(segm_input_dim=(256,256), segm_inter_dim=(256,256),
                        backbone_pretrained=True, topk_pos=3, topk_neg=3, mixer_channels=2):
    # backbone
    backbone_net = backbones.resnet50(pretrained=backbone_pretrained)

    # segmentation dimensions
    segm_input_dim = (64, 256, 512, 1024)
    segm_inter_dim = (4, 16, 32, 64)
    segm_dim = (64, 64)  # convolutions before cosine similarity

    # segmentation
    segm_predictor = segmmodels.D3S_Attn02_DW_RGBD()

    net = DepthSegmNet(feature_extractor=backbone_net, segm_predictor=segm_predictor,
                       segm_layers=['conv1', 'layer1', 'layer2', 'layer3'], extractor_grad=False)  # extractor_grad=False

    return net

@model_constructor
def depth_segm_D3S_DWSim_MP_resnet50(segm_input_dim=(256,256), segm_inter_dim=(256,256),
                        backbone_pretrained=True, topk_pos=3, topk_neg=3, mixer_channels=2):
    # backbone
    backbone_net = backbones.resnet50(pretrained=backbone_pretrained)

    # segmentation dimensions
    segm_input_dim = (64, 256, 512, 1024)
    segm_inter_dim = (4, 16, 32, 64)
    segm_dim = (64, 64)  # convolutions before cosine similarity

    # segmentation
    segm_predictor = segmmodels.D3S_DWSim_RGBD()

    net = DepthSegmNet(feature_extractor=backbone_net, segm_predictor=segm_predictor,
                       segm_layers=['conv1', 'layer1', 'layer2', 'layer3'], extractor_grad=False)  # extractor_grad=False

    return net


@model_constructor
def depth_segm_D3S_DIF_MP_resnet50(segm_input_dim=(256,256), segm_inter_dim=(256,256),
                        backbone_pretrained=True, topk_pos=3, topk_neg=3, mixer_channels=2):
    # backbone
    backbone_net = backbones.resnet50(pretrained=backbone_pretrained)

    # segmentation dimensions
    segm_input_dim = (64, 256, 512, 1024)
    segm_inter_dim = (4, 16, 32, 64)
    segm_dim = (64, 64)  # convolutions before cosine similarity

    # segmentation
    segm_predictor = segmmodels.D3S_DIF_RGBD()

    net = DepthSegmNet(feature_extractor=backbone_net, segm_predictor=segm_predictor,
                       segm_layers=['conv1', 'layer1', 'layer2', 'layer3'], extractor_grad=False)  # extractor_grad=False

    return net

@model_constructor
def depth_segm_D3S_DIF02_MP_resnet50(segm_input_dim=(256,256), segm_inter_dim=(256,256),
                        backbone_pretrained=True, topk_pos=3, topk_neg=3, mixer_channels=2):
    # backbone
    backbone_net = backbones.resnet50(pretrained=backbone_pretrained)

    # segmentation dimensions
    segm_input_dim = (64, 256, 512, 1024)
    segm_inter_dim = (4, 16, 32, 64)
    segm_dim = (64, 64)  # convolutions before cosine similarity

    # segmentation
    segm_predictor = segmmodels.D3S_DIF02_RGBD()

    net = DepthSegmNet(feature_extractor=backbone_net, segm_predictor=segm_predictor,
                       segm_layers=['conv1', 'layer1', 'layer2', 'layer3'], extractor_grad=False)  # extractor_grad=False

    return net

@model_constructor
def depth_segm_D3S_DIF03_MP_resnet50(segm_input_dim=(256,256), segm_inter_dim=(256,256),
                        backbone_pretrained=True, topk_pos=3, topk_neg=3, mixer_channels=2):
    # backbone
    backbone_net = backbones.resnet50(pretrained=backbone_pretrained)

    # segmentation dimensions
    segm_input_dim = (64, 256, 512, 1024)
    segm_inter_dim = (4, 16, 32, 64)
    segm_dim = (64, 64)  # convolutions before cosine similarity

    # segmentation
    segm_predictor = segmmodels.D3S_DIF03_RGBD()

    net = DepthSegmNet(feature_extractor=backbone_net, segm_predictor=segm_predictor,
                       segm_layers=['conv1', 'layer1', 'layer2', 'layer3'], extractor_grad=False)  # extractor_grad=False

    return net

@model_constructor
def depth_segm_D3S_DIF04_MP_resnet50(segm_input_dim=(256,256), segm_inter_dim=(256,256),
                        backbone_pretrained=True, topk_pos=3, topk_neg=3, mixer_channels=2):
    # backbone
    backbone_net = backbones.resnet50(pretrained=backbone_pretrained)

    # segmentation dimensions
    segm_input_dim = (64, 256, 512, 1024)
    segm_inter_dim = (4, 16, 32, 64)
    segm_dim = (64, 64)  # convolutions before cosine similarity

    # segmentation
    segm_predictor = segmmodels.D3S_DIF04_RGBD()

    net = DepthSegmNet(feature_extractor=backbone_net, segm_predictor=segm_predictor,
                       segm_layers=['conv1', 'layer1', 'layer2', 'layer3'], extractor_grad=False)  # extractor_grad=False

    return net

@model_constructor
def depth_segm_D3S_DIF05_MP_resnet50(segm_input_dim=(256,256), segm_inter_dim=(256,256),
                        backbone_pretrained=True, topk_pos=3, topk_neg=3, mixer_channels=2):
    # backbone
    backbone_net = backbones.resnet50(pretrained=backbone_pretrained)

    # segmentation dimensions
    segm_input_dim = (64, 256, 512, 1024)
    segm_inter_dim = (4, 16, 32, 64)
    segm_dim = (64, 64)  # convolutions before cosine similarity

    # segmentation
    segm_predictor = segmmodels.D3S_DIF05_RGBD()

    net = DepthSegmNet(feature_extractor=backbone_net, segm_predictor=segm_predictor,
                       segm_layers=['conv1', 'layer1', 'layer2', 'layer3'], extractor_grad=False)  # extractor_grad=False

    return net


@model_constructor
def depth_segm_D3S_BBS_MP_resnet50(segm_input_dim=(256,256), segm_inter_dim=(256,256),
                        backbone_pretrained=True, topk_pos=3, topk_neg=3, mixer_channels=2):
    # backbone
    backbone_net = backbones.resnet50(pretrained=backbone_pretrained)

    # segmentation dimensions
    segm_input_dim = (64, 256, 512, 1024)
    segm_inter_dim = (4, 16, 32, 64)
    segm_dim = (64, 64)  # convolutions before cosine similarity

    # segmentation
    segm_predictor = segmmodels.D3S_BBS_RGBD()

    net = DepthSegmNet(feature_extractor=backbone_net, segm_predictor=segm_predictor,
                       segm_layers=['conv1', 'layer1', 'layer2', 'layer3'], extractor_grad=False)  # extractor_grad=False

    return net
