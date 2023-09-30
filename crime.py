from flask import Flask, render_template, request, flash, redirect, session, url_for
from matplotlib import pyplot
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from wtforms import Form, StringField, PasswordField, TextAreaField, SubmitField, BooleanField, validators
from wtforms import EmailField
from wtforms.validators import DataRequired
from werkzeug.utils import secure_filename
from passlib.hash import sha256_crypt
from functools import wraps
from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.exc import IntegrityError
from flask_mail import Mail, Message
from sqlalchemy import Column, Integer, Float, ForeignKey, String
from sqlalchemy.orm import relationship
import random
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os

UPLOAD_FOLDER = 'static/profile_pic'
ALLOWED_EXTENSIONS = set(['gif', 'jpeg', 'jpg', 'png'])

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///your_sqlite_database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = 'your_secret_key'  # Replace with a secure secret key


db = SQLAlchemy(app)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///your_sqlite_database.db'



# Define the User model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    addhar = db.Column(db.String(12), unique=True, nullable=False)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    mobile = db.Column(db.String(10), nullable=False)
    password = db.Column(db.String(100), nullable=False)
    date_register = db.Column(db.String(20), nullable=False)
    ip_address = db.Column(db.String(20))
    status = db.Column(db.Integer, default=0)

def __init__(self, addhar, name, email, mobile, password, date_register, ip_address):
    self.addhar = addhar
    self.name = name
    self.email = email
    self.mobile = mobile
    self.password = password
    self.date_register = date_register
    self.ip_address = ip_address

    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///your_sqlite_database.db'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Create the application context before creating the tables
with app.app_context():
# Create the database tables
    db.create_all()



@app.route('/')
def index():
    return render_template('home.html')


    
def init_db():
    with app.app_context():
        db = get_db()
        with app.open_resource('schema.sql', mode='r') as f:
            db.cursor().executescript(f.read())
        db.commit()

def get_db():
    if 'db' not in g:
        g.db = sqlite3.connect(
            app.config['SQLALCHEMY_DATABASE_URI'],
            detect_types=sqlite3.PARSE_DECLTYPES
        )
        g.db.row_factory = sqlite3.Row
    return g.db


    #creating Registration Form
# Registration Form - Removed Email Validation
class RegistrationForm(Form):
    addhar = StringField('Addhar', [validators.DataRequired("Please Enter your Addhar Number"), validators.Regexp(regex=r'\d{12}$', message="Addhar must be of 12 Digits")])
    name = StringField('Name', [validators.Length(min=1, max=100)])
    email = EmailField('Email')
    mobile = StringField('Mobile', [validators.DataRequired("Please Enter your Mobile Number"), validators.Regexp(regex=r'\d{10}$', message="Mobile Number must be of 10 Digits")])
    password = PasswordField('Password', [
        validators.Length(min=8, max=50),
        validators.DataRequired("Please Enter Password"),
        validators.EqualTo('confirm_password', message="Password does not match")
    ])
    confirm_password = PasswordField('Confirm Password')
    accept_tos = BooleanField('I accept the Terms of Service and Privacy Notice', [DataRequired()])

# Registration Route - Removed Email and Addhar Check
@app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegistrationForm(request.form)
    if request.method == 'POST' and form.validate():
        addhar = form.addhar.data
        mobile = form.mobile.data
        name = form.name.data
        email = form.email.data  # No email validation
        password = sha256_crypt.encrypt(str(form.password.data))
        timestamp = int(time.mktime(datetime.utcnow().timetuple()))
        date_register = str(timestamp)
        ip = request.environ.get('HTTP_X_REAL_IP', request.remote_addr)

        # Insert the user data into the database without checking email and addhar
        new_user = User(name=name, email=email, mobile=mobile, addhar=addhar, password=password, date_register=date_register, ip_address=ip)
        db.session.add(new_user)
        db.session.commit()
        flash('Registration successful! You can now log in.', 'success')
        return redirect(url_for('login'))

    return render_template('register.html', form=form)

# Login Route - No Email Verification
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        # Getting the form fields data
        email_candidate = request.form['email']
        password_candidate = request.form['password']

        # Fetching the user data
        user = User.query.filter_by(email=email_candidate).first()

        if user:
            # Fetching the password from the database
            password_db = user.password

            # Comparing passwords
            if sha256_crypt.verify(password_candidate, password_db):
                session['logged_in'] = True
                session['username'] = user.name
                session['userid'] = user.id

                flash("You are now logged in", 'success')
                return redirect(url_for('index'))
            else:
                error = "Invalid login credentials"
                return render_template('login.html', error=error)
        else:
            error = "User does not exist"
            return render_template('login.html', error=error)

    return render_template('login.html')




#checking if logged_in or not
def login_required(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        if 'logged_in' in  session:
            return f(*args, **kwargs)
        else:
            flash("Unauthorized User",'danger')
            return redirect(url_for('login'))
    return wrap



#creating dashboards and its logics
@app.route('/dashboard', methods=['GET',"POST"])
@login_required
def dashboard():
    if 'logged_in' in  session:
        return render_template('dashboard.html')
    else:
        return redirect(url_for('login'))




@app.route('/dashboard_users', methods=['POST', 'GET'])
@login_required
def dashboard_users():
    users = User.query.all()
    
    return render_template('dashboard_users.html', users=users)





@app.route('/dashboard_crimes', methods=['GET', 'POST'])
@login_required
def dashboard_crimes():
    if 'logged_in' in session:
        # Fetch crimes with crime_status=1
        crime_data = Crime.query.filter_by(crime_status=1).order_by(Crime.crime_id.asc()).all()

        # Fetch crimes with crime_status=0
        crime_deleted = Crime.query.filter_by(crime_status=0).order_by(Crime.crime_id.asc()).all()

        return render_template('dashboard_crimes.html', crime_data=crime_data, crime_deleted=crime_deleted)
    else:
        return redirect(url_for('login'))


@app.route('/add_crime', methods=['GET', 'POST'])
@login_required
def add_crime():
    if request.method == 'POST':
        crime_type = request.form['crime_type']

        # Check if the crime type already exists
        existing_crime = Crime.query.filter_by(crime_type=crime_type).first()

        if not existing_crime:
            # Create a new Crime instance and add it to the database
            new_crime = Crime(crime_type=crime_type)
            db.session.add(new_crime)
            db.session.commit()

            flash("Crime Added Successfully", 'success')
            return redirect(url_for('dashboard_crimes'))
        else:
            flash("Duplicate Entry for an already existing crime", 'danger')

    return render_template('add_crime.html')



class Crime(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    crime_type = db.Column(db.String(255))
    # Add other columns as needed

# Define the State model
class State(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(255))
    # Add other columns as needed

# Define the Victim model
class Victim(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    victim_name = db.Column(db.String(255))
    # Add other columns as needed

# Define the Criminal model
class Criminal(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    criminal_name = db.Column(db.String(255))

with app.app_context():
    db.create_all()

@app.route('/dashboard_crimes_records',methods=['GET','POST'])
@login_required
def dashboard_crimes_records():
    # Fetching the crime record from the database
    crime_data = Crime.query.all()

    # Fetching the state records from the database
    state_data = State.query.all()

    if request.method == "POST":
        # Retrieve form data
        victim_name = request.form['victim_name']
        victim_father_name = request.form['victim_father_name']
        victim_age = request.form['victim_age']
        victim_gender = request.form['victim_gender']
        victim_address = request.form['victim_address']
        victim_state = request.form['victim_state']
        victim_district = request.form['victim_district']
        victim_police_station = request.form['victim_police_station']

        criminal_name = request.form['criminal_name']
        criminal_father_name = request.form['criminal_father_name']
        criminal_age = request.form['criminal_age']
        criminal_gender = request.form['criminal_gender']
        criminal_address = request.form['criminal_address']
        criminal_state = request.form['criminal_state']
        criminal_district = request.form['criminal_district']
        criminal_police_station = request.form['criminal_police_station']

        crime_type = request.form['crime_type']
        crime_location = request.form['crime_location']
        happen_when = request.form['happened_when']
        crime_state = request.form['crime_state']
        crime_district = request.form['crime_district']
        crime_police_station = request.form['crime_police_station']

        # Create and add records to the respective tables in the database
        try:
            new_victim = Victim(victim_name, victim_father_name, victim_age, victim_gender, victim_address, victim_state, victim_district, victim_police_station)
            db.session.add(new_victim)

            new_criminal = Criminal(criminal_name, criminal_father_name, criminal_age, criminal_gender, criminal_address, criminal_state, criminal_district, criminal_police_station)
            db.session.add(new_criminal)

            new_crime = Crime(crime_type, crime_location, happen_when, crime_state, crime_district, crime_police_station, new_victim, new_criminal)
            db.session.add(new_crime)

            db.session.commit()
            flash("Data Inserted Successfully", "success")
        except Exception as e:
            flash(str(e), 'danger')

    return render_template('dashboard_crimes_records.html', crime_data=crime_data, state_data=state_data)



class District(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    state_id = db.Column(db.Integer, db.ForeignKey('state.id'), nullable=False)
    # Add other fields as needed

class PoliceStation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    district_id = db.Column(db.Integer, db.ForeignKey('district.id'), nullable=False)

@app.route('/show_district/<int:state_id>', methods=["POST", "GET"])
@login_required
def show_district(state_id):
    if request.method == "POST":
        state = State.query.get(state_id)
        if state:
            dist_data = District.query.filter_by(state_id=state.id).all()
            div_type = request.args.get('div_type')
        else:
            dist_data = []
            div_type = None
    return render_template("/show_district.html", dist_data=dist_data, div_type=div_type)

@app.route('/show_police_station/<int:dist_id>', methods=["POST", "GET"])
@login_required
def show_police_station(dist_id):
    if request.method == "POST":
        district = District.query.get(dist_id)
        if district:
            police_station_data = PoliceStation.query.filter_by(district_id=district.id).all()
            div_type = request.args.get('div_type')
        else:
            police_station_data = []
            div_type = None
    return render_template("/show_police_station.html", police_station_data=police_station_data, div_type=div_type)




#making a class for kmeans clustering
class kmeans_clustering():
    cluster_array=[]
    centroids=[]
    labels=[]
    size_cluster=[]
    cluster_title=""
    cluster_x_name=""
    cluster_y_name=""
    n_clusters=0
    def __init__(self,cluster_array,cluster_title,cluster_x_name,cluster_y_name):
        self.cluster_array=cluster_array
        self.cluster_title=cluster_title
        self.cluster_x_name=cluster_x_name
        self.cluster_y_name=cluster_y_name
        self.size_cluster=[]
        self.n_clusters=0
        self.cencentroids=[]
        self.labels=[]
        pyplot.close('all')

    def make_kmeans_cluster(self):
        self.n_clusters=random.randint(2,5)
        kmeans = KMeans(self.n_clusters)
        kmeans.fit(self.cluster_array)
        colors=["g.","r.","c.","m.","b."]
        self.centroids = kmeans.cluster_centers_
        self.labels=kmeans.labels_


        for k in range(len(self.cluster_array)):
            pyplot.plot(self.cluster_array[k][0],self.cluster_array[k][1],colors[self.labels[k]], markersize=10)

        pyplot.scatter(self.centroids[:, 0],self.centroids[:,1],marker="x", s=130, linewidths=5, zorder=10)
        pyplot.title(self.cluster_title)
        pyplot.xlabel(self.cluster_x_name)
        pyplot.ylabel(self.cluster_y_name)

        for i in range(self.n_clusters):
            self.size_cluster += [[len(self.ClusterIndicesNumpy(i,self.labels)),colors[i]]]
        pyplot.show()
        pyplot.close()

    def ClusterIndicesNumpy(self,clustNum, labels_array): #numpy
        return np.where(labels_array == clustNum)[0]


#class for yearwise gaph analysis
class dashboard_yearwise_graph_analysis_class():
    year=""
    crime_id_data=[]
    crime_type_data=[]
    number_times_crime=[]
    def __init__(self,year):
        self.crime_id_data=[]
        self.crime_type_data=[]
        self.number_times_crime=[]
        self.year=year

    def fetch_data_for_graph(self):
        crime_data = Crime.query.filter_by(crime_status=1).all()

        for d in crime_data:
            self.crime_id_data.append(d.crime_id)
            self.crime_type_data.append(d.crime_type)

        for i in self.crime_id_data:
            count = Crime.query.filter(db.extract('year', Crime.date_time) == self.year, Crime.crime_id == i).count()
            self.number_times_crime.append(count)



@app.route('/dashboard_yearwise_graph_analysis',methods=['POST',"GET"])
@login_required
def dashboard_yearwise_graph_analysis():
    crime_type=[]
    number_times=[]
    max_index=[]
    max_value=""
    pair={}
    year_to_graph=""
    previous_year=""
    total_crime=0
    previous_crime_type=[]
    previous_number_times=[]
    previous_max_index=[]
    previous_max_value=""
    previous_pair={}
    previous_total_crime=0
    if request.method == 'POST':
        year_to_graph=request.form['year_graph']
        previous_year=int(year_to_graph)-1
        #flash(previous_year,"danger")
        dashboard_yearwise_graph_analysis_obj = dashboard_yearwise_graph_analysis_class(year=year_to_graph)
        dashboard_yearwise_graph_analysis_obj.fetch_data_for_graph()

        previous_dashboard_yearwise_graph_analysis_obj = dashboard_yearwise_graph_analysis_class(year=previous_year)
        previous_dashboard_yearwise_graph_analysis_obj.fetch_data_for_graph()

        crime_type = dashboard_yearwise_graph_analysis_obj.crime_type_data
        number_times = dashboard_yearwise_graph_analysis_obj.number_times_crime
        max_value=max(number_times)
        pair = dict(zip(crime_type, number_times))

        previous_crime_type = previous_dashboard_yearwise_graph_analysis_obj.crime_type_data
        previous_number_times = previous_dashboard_yearwise_graph_analysis_obj.number_times_crime
        previous_max_value=max(previous_number_times)
        #flash(previous_crime_type,"information")
        #flash(previous_number_times,"success")
        previous_pair = dict(zip(previous_crime_type, previous_number_times))


        #calculating the max index
        for i,j in pair.items():
            total_crime += j
            if max_value == j:
                max_index += [i]


        #calculating the max index
        for i,j in previous_pair.items():
            previous_total_crime += j
            if previous_max_value == j:
                previous_max_index += [i]


        #flash(previous_pair,"danger")
        #flash(previous_total_crime,"success")
        #flash(pair,"danger")

        flash(dashboard_yearwise_graph_analysis_obj.year+" Graph Analysis","success")
        plot_bar_graph_crime_id_times=plot_bar_graph(x=dashboard_yearwise_graph_analysis_obj.crime_type_data,y=dashboard_yearwise_graph_analysis_obj.number_times_crime,graph_title=year_to_graph+" Graph Analysis",graph_xlabel="Crime Type",graph_ylabel="Number of times crime Happened",width=0.5)
        plot_bar_graph_crime_id_times.make_plot()

    return render_template('dashboard_yearwise_graph_analysis.html',year=year_to_graph,previous_year=previous_year,crime_type_data=crime_type,number_times_crime=number_times,max_index=max_index,max_value=max_value,pair=pair.items(),total_crime=total_crime,previous_max_index=previous_max_index,previous_max_value=previous_max_value,previous_pair=previous_pair.items(),previous_total_crime=previous_total_crime)




#class for the crimewise graph analysis
class dashboard_crimewise_graph_analysis_class():
    crime_id=""
    crime_name=""
    number_times_crime=[]
    year=[1990,1991,1992,1993,1994,1995,1996,1997,1998,1999,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018]
    def __init__(self,crime_id):
        self.crime_id=crime_id
        self.crime_name=""
        self.number_times_crime=[]

def fetch_data_for_graph(self):
    # Fetch the crime type based on crime_id
    crimename = Crime.query.filter_by(crime_id=self.crime_id).first()
    self.crime_name = crimename.crime_type

    for i in self.year:
        # Count the number of times the crime occurred for each year
        count = db.session.query(Crime).filter(Crime.crime_id == self.crime_id, db.extract('year', Crime.date_time) == i).count()
        self.number_times_crime.append(count)



class dashboard_crimewise_year_range_data():
    crime_id=""
    crime_name=""
    number_times_crime=[]
    start_year=""
    end_year=""
    year=[]

    def __init__(self,start_year,end_year,crime_id):
        self.start_year=start_year
        self.end_year=end_year
        self.crime_id=crime_id
        self.crime_name=""
        self.number_times_crime=[]
        self.year=[]



    def fetch_required_data(self):
    # Fetch the crime type based on crime_id
        crimename = Crime.query.filter_by(crime_id=self.crime_id).first()
        self.crime_name = crimename.crime_type

        self.year = list(range(self.start_year, self.end_year + 1))

        for i in self.year:
            # Count the number of times the crime occurred for each year
            count = db.session.query(Crime).filter(Crime.crime_id == self.crime_id, db.extract('year', Crime.date_time) == i).count()
            self.number_times_crime.append(count)









@app.route('/dashboard_crimewise_graph_analysis', methods=['POST', 'GET'])
@login_required
def dashboard_crimewise_graph_analysis():
    crime_name = ""
    total_crime = 0
    after_total_crime = 0
    pair = {}
    after_pair = {}
    max_value = ""
    after_max_value = ""
    max_index = []
    after_max_index = []

    if request.method == 'POST':
        crime_id = request.form['crime_id_graph']
        crime = Crime.query.get(crime_id)

        if crime:
            crime_name = crime.crime_type
            flash(crime_name, "success")

            plot_bar = plot_bar_graph(x=dashboard_crimewise_graph_analysis_obj.year, y=dashboard_crimewise_graph_analysis_obj.number_times_crime, graph_title=crime_name + " Graph analysis", graph_xlabel="Year", graph_ylabel="Number of times crime happened", width=0.5)
            plot_bar.make_plot()

            # Calculate data for the year range
            dashboard_crimewise_year_range_data_obj = dashboard_crimewise_year_range_data(start_year=2001, end_year=2009, crime_id=crime_id)
            dashboard_crimewise_year_range_data_obj.fetch_required_data()
            max_value = max(dashboard_crimewise_year_range_data_obj.number_times_crime)
            pair = dict(zip(dashboard_crimewise_year_range_data_obj.year, dashboard_crimewise_year_range_data_obj.number_times_crime))
            for year, count in pair.items():
                total_crime += count
                if max_value == count:
                    max_index.append(year)

            # Calculate data for the after year range
            after_dashboard_crimewise_year_range_data_obj = dashboard_crimewise_year_range_data(start_year=2010, end_year=2018, crime_id=crime_id)
            after_dashboard_crimewise_year_range_data_obj.fetch_required_data()
            after_max_value = max(after_dashboard_crimewise_year_range_data_obj.number_times_crime)
            after_pair = dict(zip(after_dashboard_crimewise_year_range_data_obj.year, after_dashboard_crimewise_year_range_data_obj.number_times_crime))
            for year, count in after_pair.items():
                after_total_crime += count
                if after_max_value == count:
                    after_max_index.append(year)

    return render_template('dashboard_crimewise_graph_analysis.html', crime_name=crime_name, total_crime=total_crime, pair=pair.items(), max_value=max_value, after_total_crime=after_total_crime, after_pair=after_pair.items(), after_max_value=after_max_value)

#class for vaiation grph plotting
class plot_variation_graph():
    x=[]
    y=[]
    graph_title=""
    graph_xlabel=""
    graph_ylabel=""
    graph_color_style=""
    def __init__(self,x,y,graph_title,graph_xlabel,graph_ylabel,graph_color_style):
        self.x=x
        self.y=y
        self.graph_title=graph_title
        self.graph_xlabel=graph_xlabel
        self.graph_ylabel=graph_ylabel
        self.graph_color_style=graph_color_style
        pyplot.close('all')

    def make_plot(self):
        index=np.arange(len(self.x))
        pyplot.plot(index,self.y,self.graph_color_style)
        pyplot.title(self.graph_title)
        pyplot.xlabel(self.graph_xlabel)
        pyplot.ylabel(self.graph_ylabel)
        pyplot.xticks(index,self.x,fontsize=10,rotation=30)
        pyplot.show()
        pyplot.close()



#class for Bar grph plotting
class plot_bar_graph():
    x=[]
    y=[]
    graph_title=""
    graph_xlabel=""
    graph_ylabel=""
    width=""

    def __init__(self,x,y,graph_title,graph_xlabel,graph_ylabel,width):
        self.x=x
        self.y=y
        self.graph_title=graph_title
        self.graph_xlabel=graph_xlabel
        self.graph_ylabel=graph_ylabel
        self.width=width
        pyplot.close('all')

    def make_plot(self):
        index=np.arange(len(self.x))
        pyplot.bar(index,self.y,self.width)
        pyplot.title(self.graph_title)
        pyplot.xlabel(self.graph_xlabel)
        pyplot.ylabel(self.graph_ylabel)
        pyplot.xticks(index,self.x,fontsize=10,rotation=30)
        pyplot.show()
        pyplot.close()



#class for Pi grph plotting
class plot_pi_graph():
    labels=[]
    sizes=[]
    graph_title=""
    explode=[]


    def __init__(self,labels,sizes,graph_title):
        self.labels=labels
        self.sizes=sizes
        self.graph_title=graph_title
        self.explode=[]
        pyplot.close('all')

    def make_plot(self):
        for i in range(len(self.sizes)):
            self.explode.append(0.1)
        self.explode=tuple(self.explode)
        # Plot
        pyplot.pie(self.sizes,labels=self.labels,explode=self.explode,autopct='%1.1f%%', shadow=True, startangle=140)
        pyplot.axis('equal')
        pyplot.title(self.graph_title)
        pyplot.show()
        pyplot.close()


class fetch_id_age_times(db.Model):
    __tablename__ = 'Crimes1'

    crime_id = Column(Integer, primary_key=True)
    crime_type = Column(String(255), nullable=False)
    crime_status = Column(Integer, default=1)

    def __init__(self):
        self.crime_type_data = []
        self.crime_id_data = []
        self.criminal_age_data = []
        self.cluster_array_id_age = []
        self.crime_number_count = []
        self.cluster_array_crime_number_count = []

    def fetch_data(self):
        # Fetch crime data using SQLAlchemy
        crime_data = db.session.query(fetch_id_age_times).filter_by(crime_status=1).all()

        for crime in crime_data:
            self.crime_id_data.append(crime.crime_id)
            self.crime_type_data.append(crime.crime_type)

        for crime_id in self.crime_id_data:
            avg_age = db.session.query(func.avg(CrimeTable.age)).filter(CrimeTable.crime_id == crime_id).first()[0]
            if avg_age is not None:
                self.criminal_age_data.append(avg_age)
            else:
                self.criminal_age_data.append(0.0)

        for i in range(len(self.crime_id_data)):
            self.cluster_array_id_age.append([self.crime_id_data[i], self.criminal_age_data[i]])

# Create the database and tables
with app.app_context():
    db.create_all()

#making graph analysis
@app.route('/dashboard_graph_analysis',methods=["POST","GET"])
@login_required
def dashboard_graph_analysis():
    centroids=""
    size_cluster_color=[]
    length_cluster=0
    crime_data_cluster=fetch_id_age_times()
    crime_data_cluster.fetch_data()
    if request.method == 'POST':
        if request.form['submit']== "See Graph Crime Type And Criminal Age":
            class_kmeans_id_age =kmeans_clustering(cluster_array=crime_data_cluster.cluster_array_id_age,cluster_title="Criminal age and Crime Type",cluster_x_name="Crime Type",cluster_y_name="Criminal Age")
            class_kmeans_id_age.make_kmeans_cluster()

            #flash(class_kmeans_id_age.centroids,"danger")
            #flash(class_kmeans_id_age.size_cluster)

            centroids=class_kmeans_id_age.centroids
            size_cluster_color=class_kmeans_id_age.size_cluster
            length_cluster=class_kmeans_id_age.n_clusters
        if request.form['submit']== "See Graph Crime Type And Number of Times":
            class_kmeans_number_crime =kmeans_clustering(cluster_array=crime_data_cluster.cluster_array_crime_number_count,cluster_title="Number of crime happened",cluster_x_name="Crime Type",cluster_y_name="Number of times crime happened")
            class_kmeans_number_crime.make_kmeans_cluster()

            #flash(class_kmeans_number_crime.centroids,"danger")
            #flash(class_kmeans_number_crime.size_cluster)
            #flash(class_kmeans_number_crime.labels)

            centroids=class_kmeans_number_crime.centroids
            size_cluster_color=class_kmeans_number_crime.size_cluster
            length_cluster=class_kmeans_number_crime.n_clusters
    return render_template("/dashboard_graph_analysis.html",crime_type_data=crime_data_cluster.crime_type_data,crime_id_data=crime_data_cluster.crime_id_data,centroids=centroids,size_cluster_color=size_cluster_color,length_cluster=length_cluster)


@app.route('/dashboard_variation_graph',methods=['POST','GET'])
@login_required
def dashboard_variation_graph():
    first_criminal_age_under18=0
    first_criminal_age_18_25=0
    first_criminal_age_beyond_25=0
    first_total_crime=0
    first_crime_count={}
    year1=""
    second_criminal_age_under18=0
    second_criminal_age_18_25=0
    second_criminal_age_beyond_25=0
    second_total_crime=0
    second_crime_count={}
    year2=""
    max1=0
    max2=0


    crime_data=fetch_id_age_times()
    crime_data.fetch_data()

    if request.method == 'POST':
        if request.form['submit']== "See Graph Crime Type And Criminal Age" :
            variation_graph_id_age=plot_variation_graph(x=crime_data.crime_type_data,y=crime_data.criminal_age_data,graph_title="Graph Between Crime Id and Criminal Age",graph_xlabel="Crime Type",graph_ylabel="Criminal Age",graph_color_style="r--^")
            variation_graph_id_age.make_plot()

            crime_obj=fetch_data_for_age_yearwise(start_year=2009,end_year=2013)
            crime_obj.fetch_data()
            year1=crime_obj.year
            first_crime_count=crime_obj.crime_number_count

            #flash(crime_data_bar.criminal_age_data)
            for i in crime_obj.criminal_age_data:
                if i>0 and i<19:
                    first_criminal_age_under18 +=1
                elif i>18 and i<26:
                    first_criminal_age_18_25 +=1
                else:
                    first_criminal_age_beyond_25 +=1
            first_total_crime = first_criminal_age_under18 + first_criminal_age_18_25 +  first_criminal_age_beyond_25


            #analysis another half year
            crime_obj2=fetch_data_for_age_yearwise(start_year=2014,end_year=2018)
            crime_obj2.fetch_data()
            year2=crime_obj2.year

            #flash(crime_data_bar.criminal_age_data)
            for i in crime_obj2.criminal_age_data:
                if i>0 and i<19:
                    second_criminal_age_under18 +=1
                elif i>18 and i<26:
                    second_criminal_age_18_25 +=1
                else:
                    second_criminal_age_beyond_25 +=1
            second_total_crime = second_criminal_age_under18 + second_criminal_age_18_25 +  second_criminal_age_beyond_25

        if request.form['submit'] == "See Graph Crime Type And Number of Times":
            variation_graph_id_number_times=plot_variation_graph(x=crime_data.crime_type_data,y=crime_data.crime_number_count,graph_title="Graph Between Crime Id and Number of times Crime Happened",graph_xlabel="Crime Type",graph_ylabel="Number Of times Crime Happened",graph_color_style="g--*")
            variation_graph_id_number_times.make_plot()


            crime_obj=fetch_data_for_age_yearwise(start_year=2009,end_year=2013)
            crime_obj.fetch_data()
            year1=crime_obj.year
            first_crime_count=crime_obj.crime_number_count
            max1=max(first_crime_count)
            #flash(first_crime_count)
            crime_obj2=fetch_data_for_age_yearwise(start_year=2014,end_year=2018)
            crime_obj2.fetch_data()
            year2=crime_obj2.year
            second_crime_count=crime_obj2.crime_number_count
            max2=max(second_crime_count)
            #flash(second_crime_count)
            first_crime_count= dict(zip(crime_obj.crime_type_data,first_crime_count))
            #flash(first_crime_count)
            first_crime_count=first_crime_count.items()
            second_crime_count= dict(zip(crime_obj.crime_type_data,second_crime_count))
            #flash(second_crime_count)
            second_crime_count=second_crime_count.items()
    return render_template('/dashboard_variation_graph.html',first_year=year1,first_total_crime=first_total_crime,first_criminal_age_under18=first_criminal_age_under18,first_criminal_age_18_25=first_criminal_age_18_25,first_criminal_age_beyond_25=first_criminal_age_beyond_25,second_year=year2,second_total_crime=second_total_crime,second_criminal_age_under18=second_criminal_age_under18,second_criminal_age_18_25=second_criminal_age_18_25,second_criminal_age_beyond_25=second_criminal_age_beyond_25,first_crime_count=first_crime_count,second_crime_count=second_crime_count,max1=max1,max2=max2)


class fetch_data_for_age_yearwise:
    crime_type_data=[]
    crime_id_data=[]
    criminal_age_data=[]
    year=[]
    crime_number_count=[]

    def __init__(self,start_year,end_year):
        self.crime_type_data=[]
        self.crime_id_data=[]
        self.criminal_age_data=[]
        self.year=[]
        self.crime_number_count=[]
        for i in range(start_year,end_year+1,1):
            self.year += [i]
        self.year= tuple(self.year)


def fetch_data(self):
    # Fetch crime data using SQLAlchemy
    crimes = Crime.query.filter_by(crime_status=1).all()

    for crime in crimes:
        self.crime_id_data.append(crime.crime_id)
        self.crime_type_data.append(crime.crime_type)

        # Calculate average criminal age for the current crime
        avg_age = db.session.query(func.avg(CrimeTable.age)).join(CriminalTable).\
            filter(CrimeTable.crime_id == crime.crime_id,
                   extract('year', CrimeTable.dateTime).in_(self.year)).scalar()

        if avg_age:
            self.criminal_age_data.append(avg_age)
        else:
            self.criminal_age_data.append(0.0)

        # Fetch crime counts for the current crime
        crime_count = CrimeTable.query.filter_by(crime_id=crime.crime_id).\
            filter(extract('year', CrimeTable.dateTime).in_(self.year)).count()

        self.crime_number_count.append(crime_count)



@app.route('/dashboard_bar_graph',methods=['GET','POST'])
@login_required
def dashboard_bar_graph():
    first_criminal_age_under18=0
    first_criminal_age_18_25=0
    first_criminal_age_beyond_25=0
    first_total_crime=0
    first_crime_count={}
    year1=""
    second_criminal_age_under18=0
    second_criminal_age_18_25=0
    second_criminal_age_beyond_25=0
    second_total_crime=0
    second_crime_count={}
    year2=""
    max1=0
    max2=0

    crime_data_bar=fetch_id_age_times()
    crime_data_bar.fetch_data()

    if request.method == 'POST':
        if request.form['submit'] =="See Graph Crime Type And Criminal Age":
            bar_graph_id_age=plot_bar_graph(x=crime_data_bar.crime_type_data,y=crime_data_bar.criminal_age_data,graph_title="Graph Between Crime Id and Criminal Age",graph_xlabel="Crime Type",graph_ylabel="Criminal Age",width=0.5)
            bar_graph_id_age.make_plot()

            crime_obj=fetch_data_for_age_yearwise(start_year=2009,end_year=2013)
            crime_obj.fetch_data()
            year1=crime_obj.year
            first_crime_count=crime_obj.crime_number_count

            #flash(crime_data_bar.criminal_age_data)
            for i in crime_obj.criminal_age_data:
                if i>0 and i<19:
                    first_criminal_age_under18 +=1
                elif i>18 and i<26:
                    first_criminal_age_18_25 +=1
                else:
                    first_criminal_age_beyond_25 +=1
            first_total_crime = first_criminal_age_under18 + first_criminal_age_18_25 +  first_criminal_age_beyond_25


            #analysis another half year
            crime_obj2=fetch_data_for_age_yearwise(start_year=2014,end_year=2018)
            crime_obj2.fetch_data()
            year2=crime_obj2.year

            #flash(crime_data_bar.criminal_age_data)
            for i in crime_obj2.criminal_age_data:
                if i>0 and i<19:
                    second_criminal_age_under18 +=1
                elif i>18 and i<26:
                    second_criminal_age_18_25 +=1
                else:
                    second_criminal_age_beyond_25 +=1
            second_total_crime = second_criminal_age_under18 + second_criminal_age_18_25 +  second_criminal_age_beyond_25


        if request.form['submit'] == "See Graph Crime Type And Number of Times":
            bar_graph_id_number=plot_bar_graph(x=crime_data_bar.crime_type_data,y=crime_data_bar.crime_number_count,graph_title="Graph Between Crime Id and Number of times Crime happened",graph_xlabel="Crime Type",graph_ylabel="Number of times Crime Happened",width=0.5)
            bar_graph_id_number.make_plot()


            crime_obj=fetch_data_for_age_yearwise(start_year=2009,end_year=2013)
            crime_obj.fetch_data()
            year1=crime_obj.year
            first_crime_count=crime_obj.crime_number_count
            max1=max(first_crime_count)
            #flash(first_crime_count)
            crime_obj2=fetch_data_for_age_yearwise(start_year=2014,end_year=2018)
            crime_obj2.fetch_data()
            year2=crime_obj2.year
            second_crime_count=crime_obj2.crime_number_count
            max2=max(second_crime_count)
            #flash(second_crime_count)
            first_crime_count= dict(zip(crime_obj.crime_type_data,first_crime_count))
            #flash(first_crime_count)
            first_crime_count=first_crime_count.items()
            second_crime_count= dict(zip(crime_obj.crime_type_data,second_crime_count))
            #flash(second_crime_count)
            second_crime_count=second_crime_count.items()

    return render_template('/dashboard_bar_graph.html',first_year=year1,first_total_crime=first_total_crime,first_criminal_age_under18=first_criminal_age_under18,first_criminal_age_18_25=first_criminal_age_18_25,first_criminal_age_beyond_25=first_criminal_age_beyond_25,second_year=year2,second_total_crime=second_total_crime,second_criminal_age_under18=second_criminal_age_under18,second_criminal_age_18_25=second_criminal_age_18_25,second_criminal_age_beyond_25=second_criminal_age_beyond_25,first_crime_count=first_crime_count,second_crime_count=second_crime_count,max1=max1,max2=max2)





@app.route('/dashboard_pi_graph',methods=['POST','GET'])
@login_required
def dashboard_pi_graph():
    crime_data_pi=fetch_id_age_times()
    crime_data_pi.fetch_data()
    year1=""
    first_crime_count={}
    max1=0
    sum1=0
    year2=""
    second_crime_count={}
    max2=0
    sum2=0

    if request.method == 'POST':
        pi_graph_id_number = plot_pi_graph(labels=crime_data_pi.crime_type_data,sizes=crime_data_pi.crime_number_count,graph_title="Graph Between Crime and Number of times Crime happened")
        pi_graph_id_number.make_plot()


        crime_obj=fetch_data_for_age_yearwise(start_year=2009,end_year=2013)
        crime_obj.fetch_data()
        year1=crime_obj.year
        first_crime_count=crime_obj.crime_number_count
        max1=max(first_crime_count)
        sum1=sum(first_crime_count)
        #flash(first_crime_count)
        crime_obj2=fetch_data_for_age_yearwise(start_year=2014,end_year=2018)
        crime_obj2.fetch_data()
        year2=crime_obj2.year
        second_crime_count=crime_obj2.crime_number_count
        max2=max(second_crime_count)
        sum2=sum(second_crime_count)
        #flash(second_crime_count)
        first_crime_count= dict(zip(crime_obj.crime_type_data,first_crime_count))
        #flash(first_crime_count)
        first_crime_count=first_crime_count.items()
        second_crime_count= dict(zip(crime_obj.crime_type_data,second_crime_count))
        #flash(second_crime_count)
        second_crime_count=second_crime_count.items()

    return render_template('/dashboard_pi_graph.html',year1=year1,year2=year2,first_crime_count=first_crime_count,second_crime_count=second_crime_count,max1=max1,max2=max2,sum1=sum1,sum2=sum2)















def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    user = User.query.get(session['userid'])

    if request.method == 'POST':
        # Check whether the file is selected
        if 'profile_pic' not in request.files:
            flash("No file part available", 'danger')
            return redirect(request.url)

        profile_pic = request.files['profile_pic']

        if profile_pic.filename == '':
            flash("No file selected", 'danger')
            return redirect(request.url)

        if profile_pic and allowed_file(profile_pic.filename):
            filename = secure_filename(profile_pic.filename)
            profile_pic.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            filename = 'static/profile_pic/' + filename
            user.profile_pic = filename  # Update the user's profile_pic attribute
            db.session.commit()
            return redirect(request.url)

    return render_template('/profile.html', user=user)



@app.route('/user_dashboard',methods=['GET','POST'])
@login_required
def user_dashboard():
    return render_template('/user_dashboard.html')




@app.route('/overall', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'fileToUpload' not in request.files:
            return "No file part"
        
        file = request.files['fileToUpload']
        
        if file.filename == '':
            return "No selected file"
        
        if file:
            # Read the uploaded file into a Pandas DataFrame
            df = pd.read_csv(file)
            
            # Set Seaborn style
            sns.set_style('whitegrid')


            # Drop rows with missing data
            df.dropna(inplace=True)

            # Convert dates to pandas datetime format
            df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%y %H:%M')

            # Set the DataFrame index to the Date column
            df.set_index('Date', inplace=True)

            # Sample a part of the dataset
            x = df.sample(30000)

            # Plot the distribution of primary crime types
            plt.figure(figsize=(10, 6))
            x['Primary_Type'].value_counts().plot(kind='bar')
            plt.title("Crimes by Type")
            plt.xlabel("Crime Type")
            plt.ylabel("Count")
            plt.xticks(rotation=45)
            plt.show()

            # Plot the distribution of theft types
            plt.figure(figsize=(10, 6))
            x_theft = x[x['Primary_Type'] == "THEFT"]
            x_theft['Description'].value_counts(normalize=True).plot(kind='bar')
            plt.title("Theft Types")
            plt.xlabel("Theft Type")
            plt.ylabel("Normalized Count")
            plt.xticks(rotation=45)
            plt.show()

            # Plot the distribution of battery types
            plt.figure(figsize=(10, 6))
            x_battery = x[x['Primary_Type'] == "BATTERY"]
            x_battery['Description'].value_counts(normalize=True).plot(kind='bar')
            plt.title("Battery Types")
            plt.xlabel("Battery Type")
            plt.ylabel("Normalized Count")
            plt.xticks(rotation=45)
            plt.show()

            # Plot the distribution of criminal damage types
            plt.figure(figsize=(10, 6))
            x_cd = x[x['Primary_Type'] == "CRIMINAL DAMAGE"]
            x_cd['Description'].value_counts(normalize=True).plot(kind='bar')
            plt.title("Criminal Damage Types")
            plt.xlabel("Criminal Damage Type")
            plt.ylabel("Normalized Count")
            plt.xticks(rotation=45)
            plt.show()

            # Plot the distribution of narcotics types
            plt.figure(figsize=(10, 6))
            x_narc = x[x['Primary_Type'] == "NARCOTICS"]
            x_narc['Description'].value_counts(normalize=True).plot(kind='bar')
            plt.title("Narcotics Types")
            plt.xlabel("Narcotics Type")
            plt.ylabel("Normalized Count")
            plt.xticks(rotation=45)
            plt.show()

            # Plot the distribution of arrests
            plt.figure(figsize=(6, 4))
            x['Arrest'].value_counts(normalize=True).plot(kind='bar')
            plt.title("Arrests")
            plt.xlabel("Arrest")
            plt.ylabel("Normalized Count")
            plt.show()

            # Plot the distribution of arrests per crime type
            plt.figure(figsize=(10, 6))
            arrested_crime_types = x[x['Arrest'] == True]['Primary_Type']
            arrested_crime_types.value_counts(normalize=True).plot(kind='bar')
            plt.title("Arrests per Crime Type")
            plt.xlabel("Crime Type")
            plt.ylabel("Normalized Count")
            plt.xticks(rotation=45)
            plt.show()

            # Plot the number of crimes by month of the year
            plt.figure(figsize=(10, 6))
            color = (0.2, 0.4, 0.6, 0.6)
            x.resample('M').size().plot(kind='bar', color=color)
            plt.xlabel('Month')
            plt.ylabel('Number of Crimes')
            plt.title('Number of Crimes by Month of Year')
            plt.show()

            # For demonstration, we'll return a simple message
            message = "File uploaded and analyzed successfully"

                # Include a "Go Back" button to redirect to the "overall" page
            go_back_button = '<a href="/overall">Go Back</a>'

                # Return the success message and the "Go Back" button as HTML
            return f"{message}<br/>{go_back_button}"

    return render_template('overall.html')



@app.route('/logout')
def logout():
    del session['logged_in']
    session.clear()
    flash("You are successfully logout",'success')
    return redirect(url_for('login'))

if __name__=='__main__':
    app.secret_key='$0pi@123'
    app.run(debug=True)
