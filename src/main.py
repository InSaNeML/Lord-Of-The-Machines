#import pandas and numpy
import pandas as pd
import numpy as np

#read the train.csv file
print("Reading the data files.")
df = pd.read_csv("./data/train.csv", parse_dates=['send_date'])
test = pd.read_csv("./data/test.csv", parse_dates=['send_date'])

#read the campaign_data.csv file
daf = pd.read_csv("./data/campaign_data.csv")

#extract out the month and date of sending email
#also extract out the time of sendin email
#we do not need the year from the "send_date" column as year of sending emails are the same for all
print("Extracting time data.")
#df["month"] = df.send_date.map(lambda x: x.strftime('%m'))
#df["date"] = df.send_date.map(lambda x: x.strftime('%d'))
df["hour"] = df.send_date.map(lambda x: x.strftime('%H'))
test["hour"] = test.send_date.map(lambda x: x.strftime('%H'))
#df["min"] = df.send_date.map(lambda x: x.strftime('%M'))

#remove the send_date column after the data is extracted out , we dont need it anymore
df.drop(["send_date"], axis=1, inplace=True)
test.drop(["send_date"], axis=1, inplace=True)

#design some label encoders as we don't have to be concerned with the data of the email in here
from sklearn.preprocessing import LabelEncoder

#some LabelEncoders later to be used to change text data into numerical data as our ML algo can't handle text data
le_comm = LabelEncoder()
le_body = LabelEncoder()
le_sub = LabelEncoder()
le_url = LabelEncoder()

#train the Label Encoders on the required columns
print("Encoding the text data.")
le_comm.fit_transform(daf.communication_type)
le_body.fit_transform(daf.email_body)
le_sub.fit_transform(daf.subject)
le_url.fit_transform(daf.email_url)
print("Encoding complete.")

#replace the columns with the integer values after transforming
print("Transforming text data.")
daf.communication_type = le_comm.transform(daf.communication_type)
daf.email_body = le_body.transform(daf.email_body)
daf.subject = le_sub.transform(daf.subject)
daf.email_url = le_url.transform(daf.email_url)

#merge both the dataframes on the "campaign_id" column 
print("Merging dataframes.")
final = pd.merge(df, daf, on='campaign_id')
final_test = pd.merge(test, daf, on="campaign_id")

#save this dataframe for direct future use
print("Saving the dataframe to modified_train.csv")
final.to_csv("./data/modified_train.csv", header=True, index=False)

print("Saving the dataframe to modified_test.csv")
final_test.to_csv("./data/modified_test.csv", header=True, index=False)