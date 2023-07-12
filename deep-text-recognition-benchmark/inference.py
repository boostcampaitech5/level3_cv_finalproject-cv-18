import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np

from dataset import RawDataset, AlignCollate
from utils import AttnLabelConverter
from text_recognition_model import TextRecognitionModel
from PIL import Image

# load text recognition model
def load_text_recognition_model(
    img_scale: tuple = (32, 100),  # (height, width)
    num_fiducial: int = 20,
    input_channel: int = 1,
    output_channel: int = 512,
    hidden_size: int = 256,
    character: str = "0123456789가강거경계고관광구금기김나남너노누다대더도동두등라러로루리마머명모무문미바배뱌버보부북사산서소수아악안양어연영오용우울원육이인자작저전조주중지차천초추충카타파평포하허호홀히",
    batch_max_length: int = 25,
    Transformation: str = "TPS",
    FeatureExtraction: str = "ResNet",
    SequenceModeling: str = "BiLSTM",
    Prediction: str = "Attn",
    save_model: str = None,
    device: str = "cuda",
):
    """
    text recognition model 을 불러오는 함수

    Args:
        save_model(str) : 모델 weight 가 저장된 pth 파일 경로
        device(str) : cuda or cpu
    """

    assert save_model is not None
    assert Transformation == "TPS" and FeatureExtraction == "ResNet"
    assert SequenceModeling == "BiLSTM" and Prediction == "Attn"

    converter = AttnLabelConverter(character)
    num_class = len(converter.character)

    print(f"loading pretrained model from {save_model}")

    model = TextRecognitionModel(
        Transformation,
        FeatureExtraction,
        SequenceModeling,
        Prediction,
        num_fiducial,
        img_scale,
        input_channel,
        output_channel,
        hidden_size,
        num_class,
        batch_max_length,
    )
    model = torch.nn.DataParallel(model).to(device)
    # load model
    model.load_state_dict(torch.load(save_model, map_location=device))
    print("load model..")

    return model


class ResizeNormalize(object):

    def __init__(self, size, interpolation=Image.BICUBIC):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        img = img.resize(self.size, self.interpolation)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img
    
def inference(
    model=None,
    img_array: np.array = None,
    img_scale: tuple = (32, 100),  # (height, width)
    batch_size: int = 192,
    batch_max_length: int = 25,
    character: str = "0123456789가강거경계고관광구금기김나남너노누다대더도동두등라러로루리마머명모무문미바배뱌버보부북사산서소수아악안양어연영오용우울원육이인자작저전조주중지차천초추충카타파평포하허호홀히",
    device: str = "cuda",
):
    """
    text recognition model로 단일 이미지에 대한 결과를 출력하는 함수

    Args:
        model: load_ocr_model
        img_array : img 
        device(str): cpu or cuda
    """

    assert model is not None
    assert img_array is not None

    resize = transforms.Resize((img_scale[0], img_scale[1]))
    converter = AttnLabelConverter(character)
    to_tensor = transforms.ToTensor()
    resize_norm = ResizeNormalize((img_scale[1], img_scale[0]))
    # predict
    model.eval()
    with torch.no_grad():
        img_array = resize_norm(img_array)
        image_tensor = torch.cat([img_array.unsqueeze(0)], 0)
        image = image_tensor.to(device)

        # for image_tensors, image_path_list in ocr_data_loader:
        batch_size = image_tensor.size(0)
        image = image_tensor.to(device)
        # For max length prediction
        length_for_pred = torch.IntTensor([batch_max_length] * batch_size).to(device)
        text_for_pred = (
            torch.LongTensor(batch_size, batch_max_length + 1).fill_(0).to(device)
        )

        pred = model(image, text_for_pred, is_train=False)

        # select max probabilty (greedy decoding) then decode index to character
        _, pred_index = pred.max(2)
        pred_str = converter.decode(pred_index, length_for_pred)[0]

        pred_prob = F.softmax(pred, dim=2)
        pred_max_prob, _ = pred_prob.max(dim=2)

        # if 'Attn' in Prediction:
        pred_EOS = pred_str.find("[s]")
        pred = pred_str[:pred_EOS]  # prune after "end of sentence" token ([s])
        pred_max_prob = pred_max_prob[0][:pred_EOS]
        confidence_score = pred_max_prob.cumprod(dim=0)[-1]

        return pred, confidence_score
    

if __name__ == '__main__':
    # load model
    model = load_text_recognition_model(save_model="best_accuracy.pth", device="cuda")
    # inference
    sample_img = Image.open("sample_img").convert('L')
    result, confidence_score = inference(model=model, img_array=sample_img, device="cuda")
    print(f'{result:25s}\t{confidence_score:0.4f}')