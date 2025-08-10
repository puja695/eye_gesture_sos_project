import pywhatkit
import pyautogui
import time

def send_sos_message(phone_number, message):
    print(f"Sending WhatsApp message instantly to {phone_number}")
    # Open WhatsApp Web, type message but don't send yet
    pywhatkit.sendwhatmsg_instantly(phone_number, message, wait_time=20, tab_close=False)

    # Wait enough time for WhatsApp Web to load & message to appear typed
    time.sleep(20)  

    # TODO: Update these coordinates after testing on your system.
    # You can use pyautogui.position() to get the coordinates of the WhatsApp message input box
    message_box_x = 800  
    message_box_y = 1000

    pyautogui.click(message_box_x, message_box_y)  # Click message input box to focus it
    time.sleep(1)  # Small pause to ensure click is registered
    pyautogui.press('enter')  # Press Enter to send the message
    print("SOS message sent automatically.")
