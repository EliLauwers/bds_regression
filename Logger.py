import datetime
class Logger:
    def __init__(self, path):
        self.logpath = path
        with open(self.logpath, 'a') as file:
            timestamp = str(datetime.datetime.now().date())
            file.write('\n')
            file.write(f"------ NEW LOGGER {timestamp} ".ljust(75,'-'))
            file.write('\n')

    def process(self, text_of_interest = None, chapter = None):

        timestamp = datetime.datetime.now().strftime('%H:%M:%S')
        text = f"{timestamp} - {text_of_interest}\n"
        with open(self.logpath, 'a') as file:
            file.write(text)
        print(text[:-1])

