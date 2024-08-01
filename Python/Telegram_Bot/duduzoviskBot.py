from random import random
import telebot
from token_duduzoviskBot import token

token = token()
bot = telebot.TeleBot(token)

@bot.message_handler(commands=['image'])
def send_image(msg):
	random_value = random()
	bot.send_photo(msg.chat.id, f'https://picsum.photos/200/300/?{random_value}')
    
bot.polling()
