import os
import io

import cv2
import torch
import numpy as np
import dotenv
import telebot

from lenet5 import LeNet5

dotenv.load_dotenv()
TELE_API_KEY = os.getenv('TELE_API_KEY')
MODEL_PATH = "models/lenet5_2023-08-17_17-51-43.pth"

bot = telebot.TeleBot(TELE_API_KEY)
model = LeNet5()
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()


@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    bot.reply_to(message, "Hello, I'm LeNet5 model. Send me a photo of a digit and I'll try to recognize it!")
    canvas_id = bot.send_photo(message.chat.id, open('data/canvas.png', 'rb'))
    bot.pin_chat_message(canvas_id.chat.id, canvas_id.message_id)


@bot.message_handler(content_types=['photo'])
def handle_photo(message):
    file_info = bot.get_file(message.photo[-1].file_id)
    downloaded_file = bot.download_file(file_info.file_path)
    image_bytes = io.BytesIO(downloaded_file)
    image = cv2.imdecode(np.frombuffer(image_bytes.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
    res, res_im = detection(image)
    res_im_bytes = io.BytesIO()
    res_im_bytes.write(cv2.imencode('.png', res_im)[1].tobytes())
    res_im_bytes.seek(0)
    # bot.send_photo(message.chat.id, res_im_bytes.read())
    bot.send_message(message.chat.id, res)


def edit_photo(image):
    # remove columns and rows with only black pixels
    cols_to_remove = []
    for i in range(image.shape[0]):
        if i < 70 or i > image.shape[0] - 70:
            continue
        if np.mean(image[i, :]) <= 1:
            cols_to_remove.append(i)
    rows_to_remove = []
    for i in range(image.shape[1]):
        if i < 70 or i > image.shape[1] - 70:
            continue
        if np.mean(image[:, i]) <= 1:
            rows_to_remove.append(i)

    image = np.delete(image, cols_to_remove, axis=0)
    image = np.delete(image, rows_to_remove, axis=1)

    default_size = 28
    width, height = image.shape
    if width > height:
        padding = (width - height) // 2
        image = cv2.copyMakeBorder(image, 0, 0, padding, padding, cv2.BORDER_CONSTANT, value=0)
    if width < height:
        padding = (height - width) // 2
        image = cv2.copyMakeBorder(image, padding, padding, 0, 0, cv2.BORDER_CONSTANT, value=0)

    image = cv2.resize(image, (default_size, default_size), interpolation=cv2.INTER_AREA)
    return image


def detection(original_image):
    # crop white area around digit
    original_image = cv2.bitwise_not(original_image)
    contrast_image = original_image.copy()

    contrast_image[contrast_image > 20] = 255

    # make digits thicker
    kernel = np.ones((50, 50), np.uint8)
    contrast_image_thick = cv2.dilate(contrast_image, kernel, iterations=1)
    original_image_thick = cv2.dilate(original_image, kernel, iterations=1)

    # original_image = edit_photo(original_image)
    # original_image_thick = edit_photo(original_image_thick)
    # contrast_image = edit_photo(contrast_image)
    contrast_image_thick = edit_photo(contrast_image_thick)

    # tensor_1 = torch.from_numpy(original_image.astype(np.float32)).view(1, 1, 28, 28)
    # tensor_2 = torch.from_numpy(original_image_thick.astype(np.float32)).view(1, 1, 28, 28)
    # tensor_3 = torch.from_numpy(contrast_image.astype(np.float32)).view(1, 1, 28, 28)
    tensor_4 = torch.from_numpy(contrast_image_thick.astype(np.float32)).view(1, 1, 28, 28)

    # logits_1 = model(tensor_1)
    # logits_2 = model(tensor_2)
    # logits_3 = model(tensor_3)
    logits_4 = model(tensor_4)
    # logits_concat = torch.cat((logits_1, logits_2, logits_3, logits_4), dim=1)
    # res = logits_concat.argmax(dim=1).item() % 10
    res = logits_4.argmax(dim=1).item()
    return res, contrast_image_thick


if __name__ == '__main__':
    bot.polling()
