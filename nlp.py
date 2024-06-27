import tensorflow as tf
from transformers import BertTokenizer, TFBertForQuestionAnswering
import numpy as np


# tf.keras.backend.clear_session()


tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
model = TFBertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')


def answer_question(question, text):
    # Tokenizar la pregunta y el texto
    inputs = tokenizer.encode_plus(question, text, add_special_tokens=True, return_tensors='tf')
    max_length = 512
    
    if inputs['input_ids'].shape[1] > max_length:
        inputs = {
        'input_ids': inputs['input_ids'][:, :max_length],
        'attention_mask': inputs['attention_mask'][:, :max_length]
    }
    input_ids = inputs['input_ids'].numpy()[0]

    # Obtener las predicciones de las posiciones de inicio y fin de la respuesta
    outputs = model(inputs)
    answer_start_scores = outputs.start_logits
    answer_end_scores = outputs.end_logits

    # Obtener las posiciones de inicio y fin con las puntuaciones más altas
    answer_start = tf.argmax(answer_start_scores, axis=1).numpy()[0]
    answer_end = (tf.argmax(answer_end_scores, axis=1) + 1).numpy()[0]

    # Convertir los IDs de los tokens de vuelta a palabras
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))

    return answer


text = """
Windows Install:

1.Prepare Installation Media:

Download the Windows installation media creation tool from the Microsoft website.
Use the tool to create a bootable USB drive or burn a DVD with the Windows installation files.

2.Boot from Installation Media:

Insert the USB drive or DVD into your computer.
Restart your computer and enter the BIOS or UEFI settings (usually by pressing a key like F2, F12, ESC, or DEL during startup).
In the BIOS or UEFI settings, change the boot order to prioritize the USB drive or DVD drive.
Save the changes and exit the BIOS or UEFI settings. Your computer should now boot from the installation media.

3.Install Windows:

Follow the on-screen instructions to begin the Windows installation process,
Enter your language, time and currency format, and keyboard or input method when prompted,
Click "Install Now" and enter your product key when prompted,
Choose the edition of Windows you want to install and agree to the terms and conditions,
Select a custom installation and choose the drive where you want to install Windows. You may need to format the drive before proceeding.
Follow the rest of the on-screen instructions to complete the installation process. Your computer will restart several times during the installation.

4.Complete Setup:

Once Windows is installed, you'll need to set up your user account, password, and other preferences.
Windows will also install drivers for your hardware during this process.
"""


question ="can you explain me the step 3 with details"


answer = answer_question(question, text)
print(f"Pregunta: {question}")
print(f"Respuesta: {answer}")

