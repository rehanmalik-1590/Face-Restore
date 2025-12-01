import os
import cv2
import torch
from torchvision.transforms.functional import normalize
from basicsr.utils import imwrite, img2tensor, tensor2img
from basicsr.utils.download_util import load_file_from_url
from facelib.utils.face_restoration_helper import FaceRestoreHelper
from facelib.utils.misc import is_gray
from basicsr.utils.registry import ARCH_REGISTRY


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


pretrain_model_url = {
    'restoration': 'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth',
}


if __name__ == '__main__':
    # ======= CONFIG =======
    INPUT_IMAGE = "try.jpg"  
    OUTPUT_PATH = "results"   
    FIDELITY_WEIGHT = 0.8  
    UPSCALE = 4

    device = get_device()

    # ======= LOAD IMAGE =======
    img = cv2.imread(INPUT_IMAGE, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"❌ Could not read input image at {INPUT_IMAGE}")

    # ======= CREATE OUTPUT FOLDER =======
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    # ======= LOAD CODEFORMER MODEL =======
    net = ARCH_REGISTRY.get('CodeFormer')(
        dim_embd=512, codebook_size=1024, n_head=8, n_layers=9,
        connect_list=['32', '64', '128', '256']
    ).to(device)

    ckpt_path = load_file_from_url(
        url=pretrain_model_url['restoration'],
        model_dir='weights/CodeFormer', progress=True, file_name=None
    )
    checkpoint = torch.load(ckpt_path)['params_ema']
    net.load_state_dict(checkpoint)
    net.eval()

    # ======= FACE RESTORE HELPER =======
    face_helper = FaceRestoreHelper(
        upscale_factor=UPSCALE,
        face_size=512,
        crop_ratio=(1, 1),
        det_model='retinaface_resnet50',
        save_ext='png',
        use_parse=True,
        device=device
    )

    # ======= PROCESS IMAGE =======
    face_helper.clean_all()
    face_helper.read_image(img)

    num_det_faces = face_helper.get_face_landmarks_5(only_center_face=False, resize=640, eye_dist_threshold=5)
    print(f"Detected {num_det_faces} faces.")
    face_helper.align_warp_face()

    for idx, cropped_face in enumerate(face_helper.cropped_faces):
        cropped_face_t = img2tensor(cropped_face / 255., bgr2rgb=True, float32=True)
        normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
        cropped_face_t = cropped_face_t.unsqueeze(0).to(device)

        with torch.no_grad():
            output = net(cropped_face_t, w=FIDELITY_WEIGHT, adain=True)[0]
            restored_face = tensor2img(output, rgb2bgr=True, min_max=(-1, 1))
        restored_face = restored_face.astype('uint8')
        face_helper.add_restored_face(restored_face, cropped_face)

    face_helper.get_inverse_affine(None)
    restored_img = face_helper.paste_faces_to_input_image(upsample_img=None, draw_box=False)

    # ======= SAVE OUTPUT =======
    output_file = os.path.join(OUTPUT_PATH, "restored_image.png")
    imwrite(restored_img, output_file)
    print(f"\n Restored image saved at: {output_file}")
