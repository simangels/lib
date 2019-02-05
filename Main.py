from  setting_up_csv import Setting_Up_Csv
from logistical_regression import Logistical_Regression
from testing import Testing

my_data_file = Setting_Up_Csv("logistic_reg_data.csv")
my_data_file.read_Data()
log_reg = Logistical_Regression(my_data_file)

testing = Testing(log_reg.theta ,my_data_file)

from flask import Flask
apps=Flask(__name__)
from post import app
apps.register_blueprint(app)
apps.run()