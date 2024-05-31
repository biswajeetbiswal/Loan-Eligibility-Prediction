from flask import Flask, request, render_template
from flask_cors import cross_origin
import sklearn
import pickle
import pandas as pd

app = Flask(__name__)
model = pickle.load(open("loan_rf.pkl", "rb"))

@app.route("/")
@cross_origin()
def home():
    return render_template("home.html")

@app.route("/predict", methods=["GET", "POST"])
@cross_origin()
def predict():
    if request.method == "POST":
        income_annum = request.form["Ann_Inc"]
        loan_amount = request.form["Loan_Amt"]
        cibil_score = request.form["Cibil_rat"]
        loan_term = request.form["Loan_term"]
        residential_assets_value = request.form["Res_ass"]
        commercial_assets_value = request.form["Com_ass"]
        luxury_assets_value = request.form["Lux_ass"]
        bank_asset_value = request.form["Bank_ass"]
        no_of_dependents = 2
        edu_status = 1
        type_employment = 1

        prediction = model.predict([[
            no_of_dependents,
            income_annum,
            loan_amount,
            loan_term,
            cibil_score,
            residential_assets_value,
            commercial_assets_value,
            luxury_assets_value,
            bank_asset_value,
            edu_status,
            type_employment,
        ]])

        output = prediction[0]

        if output == 0:
            text = "You are not eligible for the loan."
        else:
            text = "You are eligible for the loan."

        return render_template('home.html',
                               prediction_text=text,
                               Ann_Inc=income_annum,
                               Loan_Amt=loan_amount,
                               Loan_term=loan_term,
                               Cibil_rat=cibil_score,
                               Res_ass=residential_assets_value,
                               Com_ass=commercial_assets_value,
                               Lux_ass=luxury_assets_value,
                               Bank_ass=bank_asset_value)

    return render_template("home.html")

if __name__ == "__main__":
    app.run(debug=True)
