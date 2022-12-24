import tkinter as tk
from tkinter import messagebox
from tkinter import filedialog as fd
import tkinter.scrolledtext as ScrolledText
import os
import sys
from gru import *
from lstm import *
from transformer import *
import torch.optim as o
import logging
import threading

def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

def training_file_choose():
    train_file_text = fd.askopenfilename()
    train_file.config(text=train_file_text)

def testing_file_choose():
    test_file_text = fd.askopenfilename()
    test_file.config(text=test_file_text)

def logger(model):

    class TextHandler(logging.Handler):

        def __init__(self, text):
            logging.Handler.__init__(self)
            self.text = text

        def emit(self, record):
            msg = self.format(record)
            def append():
                self.text.configure(state='normal')
                self.text.insert(tk.END, msg + '\n')
                self.text.configure(state='disabled')
                self.text.yview(tk.END)
            self.text.after(0, append)

    class myGUI(tk.Frame):

        def __init__(self, parent, name, *args, **kwargs):
            tk.Frame.__init__(self, parent, *args, **kwargs)
            self.root = parent
            self.build_gui(name)

        def build_gui(self, name):                    
            # Build GUI
            self.root.resizable(0, 0)
            self.root.title(f'{name}')
            x = (self.root.winfo_screenwidth()/2) - 300
            y = (self.root.winfo_screenheight()/2) - 400
            self.root.geometry('%dx%d+%d+%d' % (600, 800, x, y))

            self.root.option_add('*tearOff', 'FALSE')
            self.grid(column=0, row=0, sticky='ew')
            self.grid_columnconfigure(0, weight=1, uniform='a')
            self.grid_columnconfigure(1, weight=1, uniform='a')
            self.grid_columnconfigure(2, weight=1, uniform='a')
            self.grid_columnconfigure(3, weight=1, uniform='a')

            # Add text widget to display logging info
            st = ScrolledText.ScrolledText(self, state='disabled')
            st.configure(font='TkFixedFont')
            st.grid(column=0, row=1, sticky='w', columnspan=4)

            # Create textLogger
            text_handler = TextHandler(st)

            # Logging configuration
            logging.basicConfig(filename=f'./logs/{model}.log', #make sure different models are logged to different log files.
                level=logging.INFO, 
                format='%(asctime)s - %(levelname)s - %(message)s')        

            # Add the handler to logger
            logger = logging.getLogger()        
            logger.addHandler(text_handler)

    def streaming(text):
        logging.info(text)

    def run_gru():

        try:
            train_data, train_labels = take_data_gru(train_file.cget("text"), int(timeseries_length.get()))
            test_data, test_labels = take_data_gru(test_file.cget("text"), int(timeseries_length.get()))

            if torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")

            training_time_list = []
            train_accuracy_list = []
            train_set_testing_time_list = []
            test_accuracy_list = []
            test_set_testing_time_list = []

            m = GRU(int(number_of_features.get()), int(number_of_classes.get()), int(number_of_layers.get()), True)
            m.to(device)

            optim = o.Adam(m.parameters(), lr=float(learning_rate.get()))
            lf = nn.CrossEntropyLoss()
            m, training_time = train_gru(train_data, train_labels, m, optim, lf, device, int(epoch.get()), streaming)
            training_time_list.append(training_time)

            train_acc, train_set_testing_time = test_gru(train_data, train_labels, m, device)
            train_accuracy_list.append(train_acc)
            train_set_testing_time_list.append(train_set_testing_time)

            test_acc, test_set_testing_time = test_gru(test_data, test_labels, m, device)
            test_accuracy_list.append(test_acc)
            test_set_testing_time_list.append(test_set_testing_time)

            streaming("\nStatistics:")
            streaming(f"Average Training Time                -----> {sum(training_time_list) / len(training_time_list)}")
            streaming(f"Average Training Accuracy            -----> {sum(train_accuracy_list) / len(train_accuracy_list)}")
            streaming(f"Maximum Training Accuracy            -----> {max(train_accuracy_list)}")
            streaming(f"Minimum Training Accuracy            -----> {min(train_accuracy_list)}")
            streaming(f"Average Testing Time of Training Set -----> {sum(train_set_testing_time_list) / len(train_set_testing_time_list)}")
            streaming(f"Average Testing Accuracy             -----> {sum(test_accuracy_list) / len(test_accuracy_list)}")
            streaming(f"Maximum Testing Accuracy             -----> {max(test_accuracy_list)}")
            streaming(f"Minimum Testing Accuracy             -----> {min(test_accuracy_list)}")
            streaming(f"Average Testing Time of Test Set     -----> {sum(test_set_testing_time_list) / len(test_set_testing_time_list)}")

        except Exception as e:
            messagebox.showerror(type(e).__name__, e)

    def run_lstm():

        try:

            train_data, train_labels = take_data_lstm(train_file.cget("text"), int(timeseries_length.get()))
            test_data, test_labels = take_data_lstm(test_file.cget("text"), int(timeseries_length.get()))

            if torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")

            training_time_list = []
            train_accuracy_list = []
            train_set_testing_time_list = []
            test_accuracy_list = []
            test_set_testing_time_list = []

            m = LSTM(int(number_of_features.get()), int(number_of_classes.get()), int(number_of_layers.get()), True)
            m.to(device)

            optim = o.Adam(m.parameters(), lr=float(learning_rate.get()))
            lf = nn.CrossEntropyLoss()
            m, training_time = train_lstm(train_data, train_labels, m, optim, lf, device, int(epoch.get()), streaming)
            training_time_list.append(training_time)

            train_acc, train_set_testing_time = test_lstm(train_data, train_labels, m, device)
            train_accuracy_list.append(train_acc)
            train_set_testing_time_list.append(train_set_testing_time)

            test_acc, test_set_testing_time = test_lstm(test_data, test_labels, m, device)
            test_accuracy_list.append(test_acc)
            test_set_testing_time_list.append(test_set_testing_time)


            streaming("\nStatistics:")
            streaming(f"Average Training Time                -----> {sum(training_time_list) / len(training_time_list)}")
            streaming(f"Average Training Accuracy            -----> {sum(train_accuracy_list) / len(train_accuracy_list)}")
            streaming(f"Maximum Training Accuracy            -----> {max(train_accuracy_list)}")
            streaming(f"Minimum Training Accuracy            -----> {min(train_accuracy_list)}")
            streaming(f"Average Testing Time of Training Set -----> {sum(train_set_testing_time_list) / len(train_set_testing_time_list)}")
            streaming(f"Average Testing Accuracy             -----> {sum(test_accuracy_list) / len(test_accuracy_list)}")
            streaming(f"Maximum Testing Accuracy             -----> {max(test_accuracy_list)}")
            streaming(f"Minimum Testing Accuracy             -----> {min(test_accuracy_list)}")
            streaming(f"Average Testing Time of Test Set     -----> {sum(test_set_testing_time_list) / len(test_set_testing_time_list)}")

        except Exception as e:
            messagebox.showerror(type(e).__name__, e)

    def run_transformer():

        try:

            dropout = float(0.1)

            train_data, train_labels = take_data_transformer(train_file.cget("text"), int(timeseries_length.get()))
            test_data, test_labels = take_data_transformer(test_file.cget("text"), int(timeseries_length.get()))

            training_time_list = []
            train_accuracy_list = []
            train_set_testing_time_list = []
            test_accuracy_list = []
            test_set_testing_time_list = []

            m = Transformer(int(number_of_features.get()), int(number_of_classes.get()), int(number_of_layers.get()), True, int(pos_encode_dimension.get()), dropout, int(timeseries_length.get()))
            optim = o.Adam(m.parameters(), lr=float(learning_rate.get()))
            lf = nn.CrossEntropyLoss()
            m, training_time = train_transformer(train_data, train_labels, m, optim, lf, int(epoch.get()), streaming)
            training_time_list.append(training_time)

            train_acc, train_set_testing_time = test_transformer(train_data, train_labels, m)
            train_accuracy_list.append(train_acc)
            train_set_testing_time_list.append(train_set_testing_time)

            test_acc, test_set_testing_time = test_transformer(test_data, test_labels, m)
            test_accuracy_list.append(test_acc)
            test_set_testing_time_list.append(test_set_testing_time)

            streaming("\nStatistics:")
            streaming(f"Average Training Time                -----> {sum(training_time_list) / len(training_time_list)}")
            streaming(f"Average Training Accuracy            -----> {sum(train_accuracy_list) / len(train_accuracy_list)}")
            streaming(f"Maximum Training Accuracy            -----> {max(train_accuracy_list)}")
            streaming(f"Minimum Training Accuracy            -----> {min(train_accuracy_list)}")
            streaming(f"Average Testing Time of Training Set -----> {sum(train_set_testing_time_list) / len(train_set_testing_time_list)}")
            streaming(f"Average Testing Accuracy             -----> {sum(test_accuracy_list) / len(test_accuracy_list)}")
            streaming(f"Maximum Testing Accuracy             -----> {max(test_accuracy_list)}")
            streaming(f"Minimum Testing Accuracy             -----> {min(test_accuracy_list)}")
            streaming(f"Average Testing Time of Test Set     -----> {sum(test_set_testing_time_list) / len(test_set_testing_time_list)}")

        except Exception as e:
            messagebox.showerror(type(e).__name__, e)
    
    def worker():

        if model == "GRU":
            run_gru()
        elif model == "LSTM":
            run_lstm()
        elif model == "Transformer":
            run_transformer()

    root = tk.Tk()
    myGUI(root,model)

    t1 = threading.Thread(target=worker, args=[])
    t1.start()

    root.mainloop()
    t1.join()

######################################################################################################################################


window = tk.Tk()
window.resizable(0, 0)
window.title("Time Series Classification & Regression App")
x = (window.winfo_screenwidth()/2) - 500
y = (window.winfo_screenheight()/2) - 300
window.geometry('%dx%d+%d+%d' % (1000, 600, x, y))

train_label = tk.Label(
    window, text="Please choose your training set:", font=("Arial", 16))
train_label.place(x=20, y=10)

train_file = tk.Label(window, text="", relief="sunken", font=("Arial", 10))

train_file.place(x=410, y=15)

train_file_button = tk.Button(window, text="Choose", borderwidth=5,
                            relief="raised", font=("Arial Bold", 12), command=training_file_choose)

train_file_button.place(x=330, y=5)

test_label = tk.Label(
    window, text="Please choose your testing set:", font=("Arial", 16))
test_label.place(x=20, y=60)

test_file = tk.Label(window, text="", relief="sunken", font=("Arial", 10))

test_file.place(x=410, y=65)

test_file_button = tk.Button(window, text="Choose", borderwidth=5,
                            relief="raised", font=("Arial Bold", 12), command=testing_file_choose)

test_file_button.place(x=330, y=55)

number_of_features_label = tk.Label(
    window, text="Please enter the number of features of your dataset: ", font=("Arial", 16))
number_of_features_label.place(x=20, y=110)

number_of_features = tk.Entry(window, relief="sunken", borderwidth=2, font=("Arial Bold", 12), width=5)
number_of_features.place(x=520, y=113)

number_of_classes_label = tk.Label(
    window, text="Please enter the number of classes of your dataset: ", font=("Arial", 16))
number_of_classes_label.place(x=20, y=160)

number_of_classes = tk.Entry(window, relief="sunken", borderwidth=2, font=("Arial Bold", 12), width=5)
number_of_classes.place(x=520, y=163)

number_of_layers_label = tk.Label(
    window, text="Please enter the number of layers of your dataset: ", font=("Arial", 16))
number_of_layers_label.place(x=20, y=210)

number_of_layers = tk.Entry(window, relief="sunken", borderwidth=2, font=("Arial Bold", 12), width=5)
number_of_layers.place(x=520, y=213)

timeseries_length_label = tk.Label(
    window, text="Please enter the timeseries length of your dataset: ", font=("Arial", 16))
timeseries_length_label.place(x=20, y=260)

timeseries_length = tk.Entry(window, relief="sunken", borderwidth=2, font=("Arial Bold", 12), width=5)
timeseries_length.place(x=520, y=263)

learning_rate_label = tk.Label(
    window, text="Please enter the learning rate for optimization: ", font=("Arial", 16))
learning_rate_label.place(x=20, y=310)

learning_rate = tk.Entry(window, relief="sunken", borderwidth=2, font=("Arial Bold", 12), width=5)
learning_rate.place(x=520, y=313)

epoch_label = tk.Label(
    window, text="Please enter the epoch value for training: ", font=("Arial", 16))
epoch_label.place(x=20, y=360)

epoch = tk.Entry(window, relief="sunken", borderwidth=2, font=("Arial Bold", 12), width=5)
epoch.place(x=520, y=363)

pos_encode_dimension_label = tk.Label(
    window, text="Please enter the positional encoding dimension: ", font=("Arial", 16))
pos_encode_dimension_label.place(x=20, y=410)

pos_encode_dimension = tk.Entry(window, relief="sunken", borderwidth=2, font=("Arial Bold", 12), width=5)
pos_encode_dimension.place(x=520, y=413)

pos_encode_dimension_info_label = tk.Label(
    window, text="(For Transformer only)", font=("Arial", 16))
pos_encode_dimension_info_label.place(x=580, y=410)

# tutorial_link = tk.Label(window, text="Tutorial Link", fg="blue",
#                          font=("Arial", 16, "underline"), cursor="hand2")
# tutorial_link.place(x=820, y=300)

# tutorial_link.bind("<Button-1>", lambda a: webbrowser.open_new(
#     "https://www.loom.com/share/7a30adc59d054212aeec8c289cd21dcf"))  # Change later on

logo = tk.Canvas(window, width=400, height=200)
logo.place(x=10, y=440)
imgy = tk.PhotoImage(file=resource_path("boun.png"))
logo.create_image(0, 0, anchor=tk.NW, image=imgy)

gru_button = tk.Button(window, text="Run GRU", borderwidth=8,
                            relief="raised", font=("Arial Bold", 20), command=lambda: logger("GRU"))

gru_button.place(x=200, y=480)

lstm_button = tk.Button(window, text="Run LSTM", borderwidth=8,
                            relief="raised", font=("Arial Bold", 20), command=lambda: logger("LSTM"))

lstm_button.place(x=380, y=480)

transformer_button = tk.Button(window, text="Run Transformer", borderwidth=8,
                            relief="raised", font=("Arial Bold", 20), command=lambda: logger("Transformer"))

transformer_button.place(x=580, y=480)


# def on_closing():
#     if messagebox.askokcancel("Quit", "You haven't submitted the form yet, do you still want to quit?"):
#         window.destroy()


# window.protocol("WM_DELETE_WINDOW", on_closing)


window.mainloop()
