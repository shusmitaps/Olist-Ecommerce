#!/usr/bin/env python
# coding: utf-8

# In[1]:


# jupyter notebook --NotebookApp.iopub_data_rate_limit=1.0e10


# In[2]:


# Import library yang dibutuhkan

# Library pengolahan data
import sqlite3
import pandas as pd
import numpy as np

# Library visualisasi
import matplotlib.pyplot as plt
import seaborn as sns

# warning
import warnings
warnings.filterwarnings("ignore")


# # Data Wrangling

# In[3]:


def get_result(query):
    conn = sqlite3.connect('olist.db') 
    cursor = conn.cursor()
    cursor.execute(query)
    data = cursor.fetchall()
    conn.close()
    return data

def create_df(data, columns):
    return pd.DataFrame(data=data, columns=columns).drop(['index'], axis=1)


# In[4]:


# Retriev all dataset 
olist_order_customer_dataset = create_df(get_result("SELECT * FROM olist_order_customer_dataset"), ['index', 'customer_id', 'customers_uniq_id', 'customer_zip_code_prefix', 'customer_city', 'customer_state'])


# In[5]:


# Menampilkan tabel olist_order_customer_dataset
#olist_order_customer_dataset


# In[6]:


olist_sellers_dataset = create_df(get_result("SELECT * FROM olist_sellers_dataset"), ['index', 'seller_id', 'seller_zip_code_prefix', 'seller_city', 'seller_state'])


# In[7]:


olist_order_reviews_dataset = create_df(get_result("SELECT * FROM olist_order_reviews_dataset"), ['index', 'review_id', 'order_id', 'review_score', 'review_comment_title', 'review_comment_message', 'review_answer_date', 'review_answer_timestamp'])


# In[8]:


olist_order_items_dataset = create_df(get_result("SELECT * FROM olist_order_items_dataset"), ['index', 'order_id', 'order_item_id', 'product_id', 'seller_id', 'shipping_limit_date', 'price', 'freight_value'])


# In[9]:


olist_order_payments_dataset = create_df(get_result("SELECT * FROM olist_order_payments_dataset"), ['index', 'order_id', 'payment_squential', 'payment_type', 'payment_installment', 'payment_value'])


# In[10]:


olist_order_dataset= create_df(get_result("SELECT * FROM olist_order_dataset"), ['index', 'order_id', 'customer_id', 'order_status', 'order_purchase_timestamp', 'order_approved_at','order_delivered_carrier_date','order_delivered_customer_date','order_estimated_date'])


# In[11]:


olist_products_dataset = create_df(get_result("SELECT * FROM olist_products_dataset"), ['index', 'product_id', 'product_category_name', 'product_name_length', 'product_description_length', 'product_photo_qty','product_weight_g','product_lenght_cm','product_height_cm','product_width_cm'])


# In[12]:


olist_geolocation_dataset = create_df(get_result("SELECT * FROM olist_geolocation_dataset"), ['index', 'geolocation_zip_code_prefix', 'geolocation_lat', 'geolocation_lng', 'geolocation_city', 'geolocation_state'])


# In[13]:


olist_product_category_name_translation = create_df(get_result("SELECT * FROM product_category_name_translation"), ['index', 'product_category_name','product_category_english'])


# In[14]:


# gabungkan data menggunakan fungsi merge
data_df = olist_order_items_dataset.merge(olist_order_dataset, on = "order_id")


# In[15]:


data_df = data_df.merge(olist_order_customer_dataset, on = "customer_id")


# In[16]:


data_df = data_df.merge(olist_order_payments_dataset, on = "order_id")


# In[17]:


data_df = data_df.merge(olist_products_dataset, on = "product_id")


# In[18]:


data_df = data_df.merge(olist_product_category_name_translation, on = "product_category_name")


# In[19]:


print(data_df.columns)


# In[20]:


data_df


# In[21]:


# Menghapus kolom yang tidak terpakai
data_df = data_df.drop(["order_item_id","order_delivered_carrier_date",'order_delivered_customer_date','product_category_name', "product_name_length", "product_description_length", "product_photo_qty", "product_weight_g", "product_lenght_cm", "product_height_cm", "product_width_cm", "customer_zip_code_prefix"], axis=1)


# In[22]:


data_df


# # Data Cleaning

# ## Handling Missing Values

# In[23]:


# Mendapatkan jumlah missing value tiap kolom
# dan mengurutkan dari yang terbesar ke terkecil
nan_col = data_df.isna().sum().sort_values(ascending = False)
nan_col


# In[24]:


# Mendapatkan persentase missing value tiap kolom
len_data = len(data_df)

percent_nan_col = (nan_col/len_data) * 100
percent_nan_col


# In[25]:


# Drop baris yang %Nan > 0
data_df.dropna(subset=['order_approved_at'], inplace=True)


# ## Identify & Handling Inkonsistent Format

# In[26]:


# Menampilkan informasi pada tabel data
data_df.info()


# In[27]:


#  Convert tipe data order_purchase_timestamp menjadi float
data_df['order_purchase_timestamp'] = pd.to_datetime(data_df['order_purchase_timestamp'])


# In[28]:


data_df[["order_purchase_timestamp"]].info()


# ## Identify & Handling outlier

# In[29]:


# Buat figure & axes
fig, ax = plt.subplots(figsize = (6, 3))

# Buat histogram plot price
sns.histplot(data = data_df, 
             x = "price", 
             bins = 100,
             ax = ax)

plt.show()


# - Terlihat skala dari x axis mencapai 7000. 
# - Hal ini terjadi karena terdapat data yang nilainya mendekati 7000. 
# - Hal ini bisa di validasi dengan melihat deskripsi statistik dari kolom price

# In[30]:


# Deskripsi statistik dari kolom price
data_df["price"].describe()


# In[31]:


# Cari Q1 & Q3
Q1 = data_df.price.quantile(0.25)
Q3 = data_df.price.quantile(0.75)

print(f"Q1 : {Q1:.2f}")
print(f"Q3 : {Q3:.2f}")


# In[32]:


# Cari IQR & BATAS MAXIMUM
IQR = Q3 - Q1
max_bound = Q3 + 1.5*IQR

print(f"IQR : {IQR:.2f}")
print(f"Maximum Boundary : {max_bound:.2f}")


# In[33]:


# Filter data tanpa outlier
data_df = data_df[data_df["price"] < max_bound]


# In[34]:


# Validasi hasil filter
data_df["price"].describe()


# In[35]:


# Buat figure & axes
fig, ax = plt.subplots(figsize = (6, 3))

# Buat histogram plot price
sns.histplot(data = data_df, 
             x = "price", 
             bins = 100,
             ax = ax)

plt.show()


# In[36]:


# Menampilkna data tabel data_df
data_df


# ## Handling Duplicated

# In[37]:


# Mengecek data duplikat
data_df.duplicated(keep=False)


# In[38]:


# Menyeleksi data yang dikategorikan sebagai data duplikat
data_df[data_df.duplicated(keep=False)]


# In[39]:


# Menghilangkan data duplikat
data_df = data_df.drop_duplicates(keep='first')
data_df


# # Data Manipulation

# In[40]:


# Mengecek min dan max Datetime pada data
min(data_df.order_purchase_timestamp), max(data_df.order_purchase_timestamp)


# In[41]:


# Convert menjadi day
data_df['day_order'] = data_df['order_purchase_timestamp'].dt.strftime('%A')


# In[42]:


# Convert menjadi month
data_df['month_order'] = data_df['order_purchase_timestamp'].dt.strftime('%B')


# In[43]:


# Convert menjadi day year
data_df['year_order'] = data_df['order_purchase_timestamp'].dt.strftime('%Y')


# In[44]:


data_df


# ## Apply Function

# In[45]:


# Membuat fungsi persen
def persen(x):
    return x/x.sum() *100

# Menghitung total order per hari di per tahunnya
orders_perday = data_df.groupby(['year_order', 'day_order'])['order_id'].count().unstack(level=0)
orders_perday['total_order'] = orders_perday.sum(axis=1)

# menyimpan hasil perhitungan persentase ke dalam kolom baru
orders_perday["percent_total_order"] = orders_perday[['2016', '2017', '2018']].apply(persen, axis=1).iloc[:, 0]

orders_perday


# In[46]:


# menghitung jumlah unique value pada kolom 'col1'
unique_customer_state = data_df['customer_state'].nunique()
unique_customer_city = data_df['customer_city'].nunique()

print("customer_state", unique_customer_state)
print("customer_city", unique_customer_city)


# In[47]:


# membuat tabel pivot
pivot_t= pd.pivot_table(data_df, values='order_id', index='customer_state', columns='year_order', aggfunc='count', fill_value=0)
pivot_t['total'] = pivot_t.sum(axis=1)
pivot_t = pivot_t.sort_values(by='total', ascending=False).head(10)
pivot_t


# ## EDA

# In[48]:


# Trend by Year:
trend_year=pd.DataFrame(data_df.groupby('year_order')['order_id'].count().sort_values(ascending=True)).reset_index()
plt.figure(figsize=(4,3)) # set ukuran figure
ax=sns.barplot(x='year_order',y='order_id',data=trend_year,palette=sns.set_palette(palette='viridis_r'))

#ax.ticklabel_format()
ax.set_xlabel('Year')
ax.set_ylabel('Total Order')
ax.set_title('Transaction Order by Year')


# The data is available from Sept 2016 to Sept 2018. 
# Therefore we see a very low sales for the year 2016 .
# Entire year data is available for 2017 and hence that year is on the higher side whereas sales till Sept for 2018 is plotted.
# For lack of entire data,we are unable to conclude any significant findings here.

# In[49]:


# Trend by day:
trend_day = pd.DataFrame(data_df.groupby('day_order')['order_id'].count().reset_index())
trend_day['week_order'] = pd.Categorical(trend_day['day_order'], categories=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'], ordered=True)
trend_day = trend_day.sort_values('week_order')

plt.figure(figsize=(6,3)) # set ukuran figure
ax=sns.barplot(x='day_order',y='order_id',data=trend_day,palette=sns.set_palette(palette='viridis_r'))
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')


#ax.ticklabel_format()
ax.set_xlabel('Day Order')
ax.set_ylabel('Total Order')
ax.set_title('Transaction Order by day')


# In[50]:


#week_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
data_df.groupby('day_order')['order_id'].count().reset_index().sort_values('order_id', ascending=False)
#pd.Categorical(trend_day_s['day_order'], categories=week_order, ordered=True)
#trend_day_s.sort_values('day_order')


# In[51]:


print("Total value of transaction on Monday:", round(data_df[data_df['day_order'] == 'Monday']['payment_value'].sum(), 2))
print("Total value of transaction on Tuesday:", round(data_df[data_df['day_order'] == 'Tuesday']['payment_value'].sum(), 2))
print("Total value of transaction on Wednesday:", round(data_df[data_df['day_order'] == 'Wednesday']['payment_value'].sum(), 2))
print("Total value of transaction on Thursday:", round(data_df[data_df['day_order'] == 'Thursday']['payment_value'].sum(), 2))
print("Total value of transaction on Friday:", round(data_df[data_df['day_order'] == 'Friday']['payment_value'].sum(), 2))
print("Total value of transaction on Saturday:", round(data_df[data_df['day_order'] == 'Saturday']['payment_value'].sum(), 2))
print("Total value of transaction on Sunday:", round(data_df[data_df['day_order'] == 'Sunday']['payment_value'].sum(), 2))


# Frekuensi order lebih tinggi pada hari Senin, Selasa sedangkan frekuensi order rendah pada hari Sabtu dan Minggu. Ini berarti bahwa pada weekend, orang tidak tertarik untuk belanja online hanya dari frekuensi order saja, tetapi ketika dikombinasikan dengan nilai rata-rata transaksi selama hari tersebut, ada nilai transaksi rata-rata yang relatif tinggi yang terjadi selama hari Sabtu dibandingkan dengan hari lainnya.

# In[52]:


# group data by order month and total payment per month
month_payment = data_df.groupby('month_order')['order_id'].count().reset_index(name='total_payment')

# set month order
month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']

# sort data by month
month_payment['month_order'] = pd.Categorical(month_payment['month_order'], categories=month_order, ordered=True)
month_payment = month_payment.sort_values('month_order')

# create bar chart
sns.barplot(x='month_order', y='total_payment', data=month_payment, palette=sns.set_palette(palette='viridis_r'))
plt.xlabel("Order Month")
plt.ylabel("Total Payment")
plt.title("Total Payment per Month")
plt.xticks(rotation='vertical')
plt.show()


# In[53]:


# group data by order month and total payment per month
month_order_ = data_df.groupby('month_order')['order_id'].count().reset_index(name='total_order')

# set month order
month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']

# sort data by month
month_order_['month_order'] = pd.Categorical(month_order_['month_order'], categories=month_order, ordered=True)
month_order_ = month_order_.sort_values('month_order')

# create bar chart
sns.barplot(x='month_order', y='total_order', data=month_order_, palette=sns.color_palette('viridis_r'))
plt.xlabel("Month Order")
plt.ylabel("Total order")
plt.title("Transaction per Month")
plt.xticks(rotation='vertical')
plt.show()


# In[54]:


# Group by month and year and count order_id
order_count = data_df.groupby(data_df["order_purchase_timestamp"].dt.to_period("M"))["order_id"].nunique()

# Create a line chart
fig, ax = plt.subplots(figsize=(10,4))
plt.plot(order_count.index.astype(str), order_count.values, color="blue")

# Set the chart title and axis labels
plt.title("Purchase Orders by Month and Year", fontsize=14)
plt.xlabel("Month & Year", fontsize=12)
plt.ylabel("Orders", fontsize=12)

# Rotate the x-axis labels
plt.xticks(rotation=90)

# Show the plot
plt.show()


# In[55]:


### Visualising top 10 most bought product categories:
order_count = data_df.groupby('product_category_english')['order_id'].count().reset_index(name='order_count')
top_categories = order_count.sort_values('order_count', ascending=False).head(10)

sns.barplot(x='product_category_english', y='order_count', data=top_categories, palette=sns.set_palette(palette='viridis_r'))
plt.xlabel("Product Category")
plt.ylabel("Total Number of orders")
plt.title("Top 10 most bought product categories")
plt.xticks(rotation='vertical')
plt.show()


# In[56]:


data_df.groupby('product_category_english')['order_id'].count().reset_index().sort_values('order_id', ascending=False)


# In[57]:


### Visualising top 10 most order customer_state:
state_count = data_df.groupby('customer_state')['order_id'].count().reset_index(name='state_count')
top_categories = state_count.sort_values('state_count', ascending=False).head(10)

sns.barplot(x='customer_state', y='state_count', data=top_categories, palette=sns.set_palette(palette='viridis_r'))
plt.xlabel("customer_state")
plt.ylabel("Total Number of orders")
plt.title("Top 10 most order customer_state")
#plt.xticks(rotation='vertical')
plt.show()


# In[58]:


### Visualising Total Payment per payment type:
payment_t = data_df.groupby('payment_type')['payment_value'].count().reset_index(name='payment_t')
top_categories = payment_t.sort_values('payment_t', ascending=False)

sns.barplot(x='payment_type', y='payment_t', data=top_categories, palette=sns.set_palette(palette='viridis_r'))
plt.xlabel("payment_type")
plt.ylabel("Total payment")
plt.title("Total Payment per payment type")
#plt.xticks(rotation='vertical')
plt.show()


# Sebagian besar pembeli online menggunakan kartu kredit sebagai metode pembayaran utama mereka diikuti oleh boleto. Menurut wikipedia, boleto adalah metode pembayaran di Brasil yang diatur oleh FEBRABAN, singkatan dari Federasi Bank Brasil. Boleto dapat dibayar melalui ATM, fasilitas cabang, dan internet banking dari setiap bank, kantor pos, agen lotere, dan beberapa supermarket hingga tanggal jatuh tempo. Setelah tanggal jatuh tempo, boleto hanya dapat dibayar melalui fasilitas bank penerbit.

# In[59]:


print("Average value of transaction on credit card:", round(data_df[data_df['payment_type'] == 'credit_card']['payment_value'].mean(), 2))
print("Average value of transaction on boleto:", round(data_df[data_df['payment_type'] == 'boleto']['payment_value'].mean(), 2))
print("Average value of transaction on voucher:", round(data_df[data_df['payment_type'] == 'voucher']['payment_value'].mean(), 2))
print("Average value of transaction on debit card:", round(data_df[data_df['payment_type'] == 'debit_card']['payment_value'].mean(), 2))


# In[60]:


### Visualising Total Payment per payment type:
status_count = data_df.groupby('order_status')['order_id'].count().reset_index(name='status_count')
top_categories = status_count.sort_values(by='order_status', ascending=True)

# menambahkan teks di atas tiap bar
for index, row in top_categories.iterrows():
    plt.annotate(str(row['status_count']), xy=(row.name, row['status_count']), ha='center', va='bottom')
    
# menambahkan label dan judul
sns.barplot(x='order_status', y='status_count', data=top_categories, palette=sns.set_palette(palette='viridis_r'), tick_label=top_categories['order_status'])
plt.xlabel("order status")
plt.ylabel("Total order")
plt.title("Order status")
plt.xticks(rotation='vertical')
plt.show()
