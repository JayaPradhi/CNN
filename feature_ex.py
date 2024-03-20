dd= '../input/utkface-new/UTKFace/'

# labels - age, gender, ethnicity
image_paths = []
age_labels = []
gender_labels = []

for i in tqdm(os.listdir(dd)):
    image_path = os.path.join(dd, i)
    temp = i.split('_')
    age = int(temp[0])
    gender = int(temp[1])
    image_paths.append(image_path)
    age_labels.append(age)
    gender_labels.append(gender)


# convert to dataframe
df = pd.DataFrame()
df['image'], df['age'], df['gender'] = image_paths, age_labels, gender_labels
df.head()



#Exploratory Data Analysis
from PIL import Image
img = Image.open(df['image'][0])
plt.axis('off')
plt.imshow(img);



sns.distplot(df['age'])


sns.countplot(df['gender'])



#Feature Extraction
def extract_features(images):
    features = []
    for image in tqdm(images):
        img = load_img(image, grayscale=True)
        img = img.resize((128, 128), Image.ANTIALIAS)
        img = np.array(img)
        features.append(img)
        
    features = np.array(features)
    # ignore this step if using RGB
    features = features.reshape(len(features), 128, 128, 1)
    return features


X = extract_features(df['image'])

X.shape
# normalize the images
X = X/255.0


y_gender = np.array(df['gender'])
y_age = np.array(df['age'])


input_shape = (128, 128, 1)



    
