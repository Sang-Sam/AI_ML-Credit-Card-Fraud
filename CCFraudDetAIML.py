
# import the necessary packages
import os
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import END
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix, recall_score, f1_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
import joblib
from keras.models import *
from keras.layers import *
import customtkinter as ctk
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier 
from ml_dtypes import *

class App:
    def __init__(self, root):
        
        ctk.set_appearance_mode("System")  
        ctk.set_default_color_theme("green")
        
        root.title("Credit Card Fraud Detection using AIML")        
        
        width=1340
        height=589
        screenwidth = root.winfo_screenwidth()
        screenheight = root.winfo_screenheight()
        alignstr = '%dx%d+%d+%d' % (width, height, (screenwidth - width) / 2, (screenheight - height) / 2)
        root.geometry(alignstr)
        root.resizable(width=False, height=False)
    
        
        root.grid_rowconfigure(0, weight=0)  
        root.grid_rowconfigure(1, weight=0) 
        root.grid_rowconfigure(2, weight=20)
        root.grid_rowconfigure(3, weight=0)
        root.grid_columnconfigure(0, weight=1)
        
        script_directory = os.path.dirname(os.path.abspath(__file__))
        
        os.chdir(script_directory)
        
        def on_closing():
            if messagebox.askokcancel("Quit", "Do you want to quit the application?"):
                root.destroy()

        def build_model():
            output_textbox.delete('0.0',END)
            output_textbox.focus_set()
            model_filename = "credit_card_fraud_model.pkl"
            
            try:
                file_path = filedialog.askopenfilename(
                    filetypes=[("CSV files", "*.csv")],  
                    title="Select a CSV file",  
                )

            except Exception as e:
                messagebox.showerror("Error", f"An error occurred: {str(e)}")
                
            try:
                filename = os.path.basename(file_path)
                dataframe = pd.read_csv(file_path)
                
                loadingpath_label.configure(text="Transaction Dataset:" + filename)
                dataframe.isnull().values.any()
                dataframe["Amount"].describe()
                
                non_fraud = len(dataframe[dataframe.Class == 0])
                fraud = len(dataframe[dataframe.Class == 1])
                fraud_percent = (fraud / (fraud + non_fraud)) * 100
                
                output_textbox.insert(tk.END,"Number of Genuine transactions: " + str(non_fraud)+ "\n")
                output_textbox.update
                output_textbox.insert(tk.END,"Number of Fraud transactions: " + str(fraud)+ "\n")
                output_textbox.update
                output_textbox.insert(tk.END, "Percentage of Fraud transactions: {:.2f}%\n".format(fraud_percent))
                output_textbox.update()
                
                update_progress(30)
                            
                scaler = StandardScaler()
                dataframe["NormalizedAmount"] = scaler.fit_transform(dataframe["Amount"].values.reshape(-1, 1))
                dataframe.drop(["Amount", "Time"], inplace= True, axis= 1)

                Y = dataframe["Class"]
                X = dataframe.drop(["Class"], axis= 1)
                
                
                (train_X, test_X, train_Y, test_Y) = train_test_split(X, Y, test_size= 0.2, random_state= 42)
                
                output_textbox.insert(tk.END,"Shape of train_X:" + str(train_X.shape)+ "\n")
                output_textbox.update()
                output_textbox.insert(tk.END,"Shape of test_X:" + str(test_X.shape)+ "\n")
                output_textbox.update()
                
                # Artificial Neural Network implementation
                ann_model = Sequential()
                ann_model.add(Dense(64, input_dim=train_X.shape[1], activation='relu'))
                ann_model.add(Dropout(0.5))
                ann_model.add(Dense(32, activation='relu'))
                ann_model.add(Dropout(0.5))
                ann_model.add(Dense(1, activation='sigmoid'))
                ann_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
                ann_model.fit(train_X, train_Y, epochs=20, batch_size=64, validation_data=(test_X, test_Y))
                
                #Decision Tree
                decision_tree = DecisionTreeClassifier()

                # Random Forest
                random_forest = RandomForestClassifier(n_estimators= 100)

                #Logistic regression
                logreg = LogisticRegression()

                #Train and Evaluate our models against the dataset
                decision_tree.fit(train_X, train_Y)
                predictions_dt = decision_tree.predict(test_X)
                decision_tree_score = decision_tree.score(test_X, test_Y) * 100

                random_forest.fit(train_X, train_Y)
                predictions_rf = random_forest.predict(test_X)
                random_forest_score = random_forest.score(test_X, test_Y) * 100

                logreg.fit(train_X,train_Y)
                predictions_lr = logreg.predict(test_X)
                logistic_regression_score = logreg.score(test_X, test_Y)*100
                
                ann_score = ann_model.evaluate(test_X, test_Y, verbose=0)
                ann_score = ann_score[1] * 100

                output_textbox.insert(tk.END,"\n Artificial Neural Network Score: " + str(ann_score)+ "\n")
                output_textbox.update()
                output_textbox.insert(tk.END,"Random Forest Score: " + str(random_forest_score)+ "\n")
                output_textbox.update()
                output_textbox.insert(tk.END,"Decision Tree Score: " + str(decision_tree_score)+ "\n")
                output_textbox.update()
                output_textbox.insert(tk.END,"Logistic Regression Score: " + str(logistic_regression_score)+ "\n")
                output_textbox.update()
                
                text = "\nThe Random Forest classifier has slightly an edge over the Decision Tree classifier and Logistic Regression Classifier. Print the metrics: accuracy, precision, recall, and f1-score."
                output_textbox.insert(tk.END, text)
                output_textbox.insert(tk.END, text="\n")
                
                confusion_matrix_dt = confusion_matrix(test_Y, predictions_dt.round())
                # Print Confusion Matrix Decision Tree
                output_textbox.insert(tk.END,text="\nConfusion Matrix Decision tree\n")
                output_textbox.insert(tk.END,text=confusion_matrix_dt)
                
                confusion_matrix_lr = confusion_matrix(test_Y, predictions_lr.round())
                # Print Confusion Matrix Logistic Regression
                output_textbox.insert(tk.END,text="\nConfusion Matrix Logistic Regression\n")
                output_textbox.insert(tk.END,text=confusion_matrix_lr)
                
                confusion_matrix_rf = confusion_matrix(test_Y, predictions_rf.round())
                # Print Confusion Matrix Logistic Regression
                output_textbox.insert(tk.END,text="\nConfusion Matrix Random Forest\n")
                output_textbox.insert(tk.END,text=confusion_matrix_rf)
                
                #Update Progress Bar
                update_progress(50)
                
                output_textbox.insert(tk.END, "\nEvaluation of Artificial Neural Network Model\n")
                output_textbox.update()
                metrics(test_Y, ann_model.predict(test_X) > 0.5)
                
                output_textbox.insert(tk.END,"\nEvaluation of Decision Tree Model\n")
                output_textbox.update()
                metrics(test_Y, predictions_dt.round())
                
                output_textbox.insert(tk.END,"\nEvaluation of Random Forest Model\n")
                output_textbox.update()
                metrics(test_Y, predictions_rf.round())

                output_textbox.insert(tk.END,"\nEvaluation of Logistic regression Model\n")
                output_textbox.update()
                metrics(test_Y, predictions_lr.round())

                #Update Progress Bar
                update_progress(70)
                
                X_resampled, Y_resampled = SMOTE().fit_resample(X, Y)
                output_textbox.insert(tk.END,"\nResampled shape of X:"+ str(X_resampled.shape) +"\n")
                output_textbox.update()
                output_textbox.insert(tk.END,"Resampled shape of Y:"+ str(Y_resampled.shape) +"\n")
                output_textbox.update()
                value_counts = Counter(Y_resampled)
                output_textbox.insert(tk.END,"Count"+ str(value_counts)+"\n")
                output_textbox.update()

                (train_X, test_X, train_Y, test_Y) = train_test_split(X_resampled, Y_resampled, test_size= 0.3, random_state= 42)

                output_textbox.insert(tk.END,"\nApply the Random Forest algorithm to our resampled data" +"\n")
                output_textbox.update()
                
                rf_resampled = RandomForestClassifier(n_estimators = 100)
                rf_resampled.fit(train_X, train_Y)

                predictions_resampled = rf_resampled.predict(test_X)
                random_forest_score_resampled = rf_resampled.score(test_X, test_Y) * 100
                output_textbox.insert(tk.END,"Random Forest Score Resampled: " + str(random_forest_score_resampled)+ "\n")
                output_textbox.update()
                
                # After training, save the model to a file
                joblib.dump(rf_resampled, model_filename)
                
                text = "\nFinal optimized Model Saved to " + model_filename + " for FRAUD detection"
                output_textbox.insert(tk.END, text)
                output_textbox.insert(tk.END, "\n")
                
                #Update Progress Bar
                update_progress(85)
                
                #Let’s visualize the predictions of our model and plot the confusion matrix.
                cm_resampled = confusion_matrix(test_Y, predictions_resampled.round())
                # Print Confusion Matrix Random Forest on Resampled Data
                output_textbox.insert(tk.END,text="\nConfusion Matrix Random Forest on Resampled Data\n")
                output_textbox.insert(tk.END,text=cm_resampled)
                
                output_textbox.insert(tk.END,"\nEvaluation of Random Forest Model\n")
                output_textbox.insert(tk.END,"Lets Verify the metrics on resampled Data\n")
                metrics(test_Y, predictions_resampled.round())
                output_textbox.update()
                
                #Update Progress Bar
                update_progress(100)
            
            except Exception as e:
                print("Error", f"An error occurred: {str(e)}") 
                
        def metrics(actuals, predictions):
            try:
                output_textbox.insert(tk.END,"Accuracy: {:.9f}".format(accuracy_score(actuals, predictions))+ "\n")
                output_textbox.update()
                output_textbox.insert(tk.END,"Precision: {:.5f}".format(precision_score(actuals, predictions))+ "\n")
                output_textbox.update()
                output_textbox.insert(tk.END,"Recall: {:.5f}".format(recall_score(actuals, predictions))+ "\n")
                output_textbox.update()
                output_textbox.insert(tk.END,"F1-score: {:.5f}".format(f1_score(actuals, predictions))+ "\n")
                output_textbox.update()
            except Exception as e:
                print("Error", f"An error occurred: {str(e)}") 
            
        def update_progress(i):
            try:    
                progress.set(i/100)
                progress_label.configure(text=f"Model Build Progress: {i}%")
            except Exception as e:
                print("Error", f"An error occurred: {str(e)}")
        
        def predict():
            output_textbox.delete('1.0',END)
            model_filename = "credit_card_fraud_model.pkl"
            
            if os.path.exists(model_filename):
                # Model file already exists, load it instead of running the model selection code
                loaded_model = joblib.load(model_filename)
                loadingpath_label.configure(text="Running Prediction...")
                transactions_scanned = 0  # Counter to track the number of transactions scanned
                correct_predictions = 0  # Counter to track the number of correct predictions

                # Infinite loop for generating random transactions and checking them
                while True:
                    transactions_scanned += 1
                    loadingpath_label.configure(text="Running Prediction: " + str(transactions_scanned)) 
                    # Generate a random transaction
                    fraudulent_data = {
                            'V1': [np.random.uniform(-5, 5)],
                            'V2': [np.random.uniform(-5, 5)],
                            'V3': [np.random.uniform(-5, 5)],
                            'V4': [np.random.uniform(-5, 5)],
                            'V5': [np.random.uniform(-5, 5)],
                            'V6': [np.random.uniform(-5, 5)],
                            'V7': [np.random.uniform(-5, 5)],
                            'V8': [np.random.uniform(-5, 5)],
                            'V9': [np.random.uniform(-5, 5)],
                            'V10': [np.random.uniform(-5, 5)],
                            'V11': [np.random.uniform(-5, 5)],
                            'V12': [np.random.uniform(-5, 5)],
                            'V13': [np.random.uniform(-5, 5)],
                            'V14': [np.random.uniform(-5, 5)],
                            'V15': [np.random.uniform(-5, 5)],
                            'V16': [np.random.uniform(-5, 5)],
                            'V17': [np.random.uniform(-5, 5)],
                            'V18': [np.random.uniform(-5, 5)],
                            'V19': [np.random.uniform(-5, 5)],
                            'V20': [np.random.uniform(-5, 5)],
                            'V21': [np.random.uniform(-5, 5)],
                            'V22': [np.random.uniform(-5, 5)],
                            'V23': [np.random.uniform(-5, 5)],
                            'V24': [np.random.uniform(-5, 5)],
                            'V25': [np.random.uniform(-5, 5)],
                            'V26': [np.random.uniform(-5, 5)],
                            'V27': [np.random.uniform(-5, 5)],
                            'V28': [np.random.uniform(-5, 5)],
                            'NormalizedAmount': [np.random.uniform(1, 1000)]
                        }

                    sample_record_df = pd.DataFrame(fraudulent_data)

                    # Predict if it's a genuine or fraud transaction
                    prediction = loaded_model.predict(sample_record_df)
                    
                    if prediction == [0]:
                        correct_predictions += 1
                        output_textbox.update()
                        
                    else:
                        
                        text = "FRAUD Transaction Detected\n"
                        
                        output_textbox.insert(tk.END, text, "green_text")
                        output_textbox.insert(tk.END, "\n")
                        
                        text = "Transactions Scanned Until Fraud:"
                        
                        output_textbox.insert(tk.END, f"{text} {transactions_scanned}"+"\n\n", "green_text")
                        loadingpath_label.configure(text="Prediction Complete..")
                        # Calculate and display accuracy
                        accuracy = correct_predictions / transactions_scanned
                        output_textbox.insert(tk.END, f"Accuracy: {accuracy:.2%}\n")
                        output_textbox.update()
                        
                        # Display details of the fraudulent transaction vertically
                        for col, value in sample_record_df.iloc[0].items():
                            output_textbox.insert(tk.END, f"{col}: {value}\n")
                            output_textbox.insert(tk.END, "\n")
                        break  # Exit the loop when a fraudulent transaction is detected
                    
                                          
        ####################
        frame_row0=ctk.CTkFrame(root,fg_color="transparent")
        frame_row0.grid(row=0,column=0,columnspan=5,padx=0, pady=1, sticky="nsew")
        frame_row0.grid_columnconfigure(0, weight=1)
        frame_row0.grid_columnconfigure(1, weight=1)
        frame_row0.grid_columnconfigure(2, weight=1)
        frame_row0.grid_columnconfigure(3, weight=1)
        frame_row0.grid_columnconfigure(4, weight=1)
        progress_label = ctk.CTkLabel(frame_row0, text="Model Build Progress: 0%")
        progress_label.grid(row=0,column=2,padx=0, pady=1, sticky="nsew")
        progress = ctk.CTkProgressBar(frame_row0, orientation="horizontal", mode="determinate")
        progress.grid(row=1,column=2)
        progress.set(0)
        ####################
        
        ####################
        frame_row1=ctk.CTkFrame(root,fg_color="transparent")
        frame_row1.grid(row=1,column=0,columnspan=5,padx=0, pady=1, sticky="nsew")
        frame_row1.grid_columnconfigure(0, weight=1)
        frame_row1.grid_columnconfigure(1, weight=1)
        frame_row1.grid_columnconfigure(2, weight=1)
        frame_row1.grid_columnconfigure(3, weight=1)
        frame_row1.grid_columnconfigure(4, weight=1)
        loadingpath_label = ctk.CTkLabel(frame_row1, text="Loading Path...")
        loadingpath_label.grid(row=0,column=0,padx=10, pady=30, sticky="nsew")
        loaddataset_btn = ctk.CTkButton(frame_row1,text="Load Dataset...",command=build_model,width=100)
        loaddataset_btn.grid(row=0,column=1,padx=10, pady=30, sticky="nsew")
        testtrx_btn = ctk.CTkButton(frame_row1,text="Click to Test Random new Transaction",command=predict,width=100)
        testtrx_btn.grid(row=0,column=2,padx=10, pady=30, sticky="nsew")
        
        frame_row2=ctk.CTkFrame(root,fg_color="transparent")
        frame_row2.grid(row=2,column=0,padx=0, pady=1, sticky="nsew")
        frame_row2.grid_columnconfigure(0, weight=1)
        frame_row2.grid_columnconfigure(1, weight=1)
        frame_row2.grid_columnconfigure(2, weight=1)
        frame_row2.grid_columnconfigure(3, weight=1)
        frame_row2.grid_columnconfigure(4, weight=1)
        output_textbox = ctk.CTkTextbox(frame_row2,wrap="word",height=450)
        output_textbox.grid(row=0,column=0,columnspan=4,rowspan=2,padx=10, pady=0, sticky="nsew")
        output_textbox.insert(tk.END,text="Program Output will appear here!")
        # Create a label with instructions
        instructions = "Welcome to the App!\n\nThis app uses AIML to detect credit card fraud transaction in real time.\n\nHere are the instructions to use the app:\n\nStep 1:\n1. First select a dataset with historical credit card transactions and load it.\n2. App will then use this dataset to identify the best suitable algorithm for high accuracy score\n3. Once the model is determined it will create a final dataset and save it as *.pkl file\n4. This file will then be used to detect any future transaction for FRAUD.\n5. Confusion Matrix are also plotted to visualize the efficiency of the various algorithms\n\nStep 2:\n1. Click on Run Prediction button, it will load the *pkl file in to the dataframe\n2. Program will then generate random transactions and run against the saved model\n3. Once fraud transaction detected the program will stop and print the fraud transaction\n\n"
        instruction_label=ctk.CTkLabel(frame_row2,text=instructions,anchor="w",justify="left",font=("calibri",13))
        instruction_label.grid(row=0,column=4,rowspan=2,padx=5, pady=0, sticky="nw")
        developer_label=ctk.CTkLabel(frame_row2,text="@Developed By: AIML TEAM",font=("calibri",9))
        developer_label.grid(row=0,column=4,rowspan=2,padx=5, pady=0, sticky="se")
       
        root.protocol("WM_DELETE_WINDOW", on_closing)

if __name__ == "__main__":
    root = ctk.CTk()
    app = App(root)
    root.mainloop()
