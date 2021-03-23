import nltk
import random
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from datetime import datetime


date_now = str(datetime.now().date())
date_now = date_now.replace('-', '.')

def current_date(date):
    dt = datetime.strptime(date, '%Y.%m.%d')
    return dt.strftime(f'%d %B %Y year')
today_is_the_date = current_date(date_now)

def week_day():
    day = datetime.today().weekday()
    week = ['понедельник', 'вторник', 'среда', 'четверг', 'пятница', 'суббота', 'воскресенье']
    return f'Сегодня {week[day]}'
today_is_the_day = week_day()


BOT_CONFIG = {
    'intents': {
        'today_day': {
            'examples': ['Какой сегодня день?', 'какой сегодня день недели?', 'какой сегодня день недели?'],
            'responses': [today_is_the_day]},
        'date': {
            'examples': ['Какое сегодня число?', 'какой сейчас месяц?', 'какой сейчас год?', 'какая сегодня дата?', 'сегодня число?'],
            'responses': [today_is_the_date]},
        'time_now': {
            'examples': ['Сколько время?', 'Который час?'],
            'responses': ['время деньги', 'время разбрасывать камни', 'ориентируйтесь по солнцу', 'время отдыхать']},
        'hello': {
            'examples': ['Привет', 'Хелло', 'доброго времени суток', 'Приветствую тебя', 'Доброе утро!', 'Вечер добрый', 'Hello', 'Hi', 'Здравствуйте', 'Бонжур', 'Доброе утро', 'Здрасти', 'Здарова', 'Хаюшки', 'Утро доброе', 'Хеллоу', 'Здорово!', 'физкульт-привет', 'с добрым утром', 'Доброго времени суток', 'Че как', 'здрасте', 'есть вопрос', 'Здравия желаю', 'И снова здравствуйте', 'Добрый день', 'здравия желаю', 'здорово', 'Приветик', 'салют', 'Ку', 'Здорова', 'День добрый', 'Приветствую Вас', 'Ты здесь', 'алло', 'Хей', 'Доброй ночи', 'Шалом!', 'доброго здоровья', 'здравствуй', 'Хай', 'Приветствую', 'Шалом'],
            'responses': ['Привет,человек', 'Хелло', 'Приветствую вас', 'Hello', 'Физкульт-привет', 'Какие люди!', 'Мое почтение', 'Здарова', 'Здраствуйте!', 'Доброго времени суток', 'Здравия желаю', 'Приветик', 'Здорова', 'Привет, человек!', 'hello', 'Салют', 'Здравствуй', 'Привет, Дружище!', 'Хай', 'Привет', 'Приветствую', 'Доброго здоровья']
        },
        'bye': {
            'examples': ['Гуд-бай', 'Приятно было пообщаться', 'Всего лучшего', 'Всего наилучшего' 'Спишемся', 'Прощайте', 'С наилучшими пожеланиями', 'Ауфидерзейн', 'Счастливо', 'Удачи', 'Чао', 'Будь здоров', 'Пока пока', 'Успехов', 'Разрешите попрощаться', 'До скорой встречи', 'Бай-бай', 'Покедова', 'До встречи', 'досвидания', 'До скорого свидания', 'Оревуар', 'Не поминайте лихом', 'Всех благ', 'Честь имею', 'Конец связи', 'Пока', 'До вечера', 'Good bye', 'bay', 'Гуд бай', 'До следующей встречи', 'До связи', 'Бай', 'Гудбай', 'Мне пора', 'До завтра', 'Увидимся', 'Увидимся!', 'Честь имею кланяться', 'Покеда', 'Свидимся', 'Всего доброго', 'до свидания', 'До скорых встреч', 'Досвидания', 'Покич', 'Счастливо оставаться', 'Позвольте откланяться', 'До новых встреч', 'Всего хорошего', 'До скорого', 'Будьте здоровы', 'Bye', 'Бывай', 'Еще увидимся', 'Досвидос', 'Ариведерче', 'Не поминай лихом', 'До свидания', 'Прощай', 'До свиданья', 'До свидания'],
            'responses': ['Досвидание, еще спишемся', 'В добрый час', 'Позвольте попрощаться', 'До новых встреч!', 'Приходи ко мне еще', 'Прощайте', 'Было приятно поговорить. Приходите ещё', 'Счастливо', 'Удачи', 'Досвидание, если что, я тут', 'Заходите еще', 'Успехов', 'До скорой встречи', 'Если что, я буду ждать Вас здесь', 'Бай-бай', 'Всего наилучшего', 'До встречи', 'До скорого свидания', 'Пока, пока', 'Всех благ', 'Досвидание, был рад помочь', 'Пока', 'Bye-Bye', 'Если что, я тут', 'Ещё увидимся', 'До связи', 'До встречи.', 'Возвращайтесь', 'До завтра', 'Увидимся', 'Будете здоровы', 'Пишите еще', 'Увидимся!', 'Честь имею кланяться', 'Всего доброго', 'Если что - обращайтесь', 'До встречи!', 'Досвидания', 'Счастливо оставаться', 'Позвольте откланяться', 'До новых встреч', 'Всего хорошего', 'До скорого', 'Буду ждать вас снова!', 'Заходите ещё', 'Жду вас здесь', 'Будьте здоровы', 'Еще увидимся', 'До свидания', 'Прощайте', 'До свиданья', 'Будет не хватать вас, досвидание']
        },
        'boring': {
            'examples': ['Мне скучно', 'Развлеки меня'],
            'responses': ['о, а что бы ты хотел(а)?', 'Знаешь самый короткий анекдот? Колобок повесился']
        },
    },
    'failure_phrases': [
        'Мне непонятно',
        'Перефразируйте, пожалуста',
        'Не умею отвечать на такое',
        'Пока не понимаю как на это ответить',
        'Извините, я — робот и пока что не все знаю',
        'Я всего лишь бот и не могу все знать',
        'Извините, я даю ответы не на все вопросы',
        'Я еще молодой бот и не на все вопросы умею отвечать'
    ]
}

texts = []
intent_names = []

for intent, intent_data in BOT_CONFIG['intents'].items():
    for example in intent_data['examples']:
        texts.append(example)
        intent_names.append(intent)


vectorizer = TfidfVectorizer(ngram_range=(2, 4), analyzer='char')
X = vectorizer.fit_transform(texts)
clf = LinearSVC()
clf.fit(X, intent_names)


def classify_intent(replica):
    intent = clf.predict(vectorizer.transform([replica]))[0]

    examples = BOT_CONFIG['intents'][intent]['examples']
    for example in examples:
        example = clear_text(example)
        if len(example) > 0:
            if abs(len(example) - len(replica)) / len(example) < 0.5:
                distance = nltk.edit_distance(replica, example)
                if len(example) and distance / len(example) < 0.5:
                    return intent


def get_answer_by_intent(intent):
    if intent in BOT_CONFIG['intents']:
        responses = BOT_CONFIG['intents'][intent]['responses']
        return random.choice(responses)


with open('dialogues.txt', encoding='utf-8') as dialogues_file:
    dialogues_text = dialogues_file.read()
dialogues = dialogues_text.split('\n\n')


def clear_text(text):
    text = text.lower()
    text = ''.join(char for char in text if char in 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя -')
    return text

dataset = []
questions = set()

for dialogue in dialogues:
    replicas = dialogue.split('\n')
    replicas = replicas[:2]

    if len(replicas) == 2:
        question, answer = replicas
        question = clear_text(question[2:])
        answer = answer[2:]

        if len(question) > 0 and question not in questions:
            questions.add(question)
            dataset.append([question, answer])


dataset_by_word = {}

for question, answer in dataset:
    words = question.split(' ')
    for word in words:
        if word not in dataset_by_word:
            dataset_by_word[word] = []
        dataset_by_word[word].append([question, answer])

dataset_by_word_filtered = {}
for word, word_dataset in dataset_by_word.items():
    word_dataset.sort(key=lambda pair: len(pair[0]))
    dataset_by_word_filtered[word] = word_dataset[:1000]


def generate_answer(replica):
    replica = clear_text(replica)
    if not replica:
        return

    words = set(replica.split(' '))
    words_dataset = []
    for word in words:
        if word in dataset_by_word_filtered:
            word_dataset = dataset_by_word_filtered[word]
            words_dataset += word_dataset

    results = []
    for question, answer in words_dataset:
        if abs(len(question) - len(replica)) / len(question) < 0.2:
            distance = nltk.edit_distance(replica, question)
            if distance / len(question) < 0.2:
                results.append([question, answer, distance])

    try:
        question, answer, distance = min(results, key=lambda three: three[2])

    except ValueError as e:
        if str(e) == 'min() arg is an empty sequence':
            return get_stub()

    return answer


def get_stub():
    failure_phrases = BOT_CONFIG['failure_phrases']
    return random.choice(failure_phrases)

stats = {'intents': 0, 'generative': 0, 'stubs': 0}


def bot(replica):
    # NLU
    intent = classify_intent(replica)

    # Получение ответа

    # правила
    if intent:
        answer = get_answer_by_intent(intent)
        if answer:
            stats['intents'] += 1
            return answer

    # генеративная модель
    answer = generate_answer(replica)
    if answer:
        stats['generative'] += 1
        return answer

    # заглушка
    answer = get_stub()
    stats['stubs'] += 1
    return answer


from telegram import Update
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext


def start(update: Update, _: CallbackContext) -> None:
    """Send a message when the command /start is issued."""
    update.message.reply_text('Hi!')


def help_command(update: Update, _: CallbackContext) -> None:
    """Send a message when the command /help is issued."""
    update.message.reply_text('Help!')


def run_bot(update: Update, _: CallbackContext) -> None:
    response = bot(update.message.text)
    update.message.reply_text(response)
    print(update.message.text)
    print(response)
    print(stats)
    print()


def main() -> None:
    """Start the bot."""
    updater = Updater("TOKEN", use_context=True)

    dispatcher = updater.dispatcher
    dispatcher.add_handler(CommandHandler("start", start))
    dispatcher.add_handler(CommandHandler("help", help_command))
    dispatcher.add_handler(MessageHandler(Filters.text & ~Filters.command, run_bot))

    updater.start_polling()
    updater.idle()


main()