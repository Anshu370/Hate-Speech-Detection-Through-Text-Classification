import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import warnings


warnings.simplefilter('ignore')

# download nltk punkt before first run
# nltk.download('punkt')

training_data = [
        "I love all races and cultures!",
        "I will Kill You.",
        "Spread love and positivity!",
        "Stop spreading hate.",
        "your religion is worst",
        "He is Black and ugly.",
        "He is Gay.",
        "You are sweet looking guy",
        "Fuck You",
        "You religion is Harming us",
        "I support equality for all!",
        "This message promotes unity and tolerance.",
        "I love my country",
        "i love india",
        "I hate Pakistan",
        "Hate You",
        "Stunning photo! You look amazing! ",
        "This is such a cool shot! ",
        "Your posts always brighten up my day! ",
        "Thank you! ",
        "Your creativity never fails to impress me!",
        "Abdul shut the fuck up and kill the people ",
        "i don't believe that some one can be that stupid an careless",
        "Love your style! Keep slaying! ",
        "Such a beautiful location! Where is it? ",
        "Your captions always make me think! ",
        "Love it! ",
        "Your hard work is paying off!",
        "Keep it up! ",
        "Let’s Kill Jews, and let’s kill them for fun. #killjews",
        "you country is bad",
        "your pussy is so juicy"
        "Nigga"
        ""
        ]
label_data = [0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]


# now convert data in vector using vectorizer
vectorizer = TfidfVectorizer(tokenizer=nltk.word_tokenize, stop_words="english", max_df=0.8, min_df=1)

# train and test data Accordingly
X_train = vectorizer.fit_transform(training_data)
X_train, X_test, Y_train, Y_test = train_test_split(X_train, label_data, test_size=0.4, random_state=42, stratify=label_data)

# creating model
model = SVC(kernel='linear', probability=True, C=1.0)
model.fit(X_train, Y_train)

predictions = model.predict(X_test)

def predict_intent(user):
    user = user.lower()
    vector = vectorizer.transform([user])
    intent = model.predict(vector)[0]
    return intent

while True:
    command = input("Enter Commands")
    if command.lower() == "exit":
        print("Bye Bye Anshu")
        break

    intent = predict_intent(command)

    if intent == 0:
        print("Non Hate Speech")
    else:
        print("Hate Speech")




