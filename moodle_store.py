# ------------------------------------------------------------------------
# Class MoodleStore
#
# Copyright   2024 Pimenko <support@pimenko.com><pimenko.com>
# Author      Jordan Kesraoui
# License     https://www.gnu.org/copyleft/gpl.html GNU GPL v3 or later
# ------------------------------------------------------------------------

import requests
import pandas as pd
from urllib.request import urlretrieve
import argparse
import os
import textract
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from transformers import BartTokenizer, BartForConditionalGeneration

class MoodleStore:

    def __init__(self, wstoken: str, wsendpoint: str, wsstorage: str):
        self.wstoken = wstoken
        self.wsendpoint = wsendpoint
        self.wsstorage = wsstorage

    def ws_get_courses_to_vectorstore(self):
        """Retrieve course information from the LMSAssistant plugin.

        Returns:
            Courses array json object.

        Raises:
            Exception with status code and response text.
        """
        data = {
            "wstoken": self.wstoken,
            "wsfunction": "local_lmsassistant_courses_assistant_info",
            "moodlewsrestformat": "json",
        }

        # HTTP request
        response = requests.get(self.wsendpoint, data)
        if response.status_code == 200:
            response = response.json()
            print("Courses needed to be stored :")
            print(response)
            return response.get('courses', [])
        else:
            raise Exception(f"Request failed with status code: {response.status_code}: {response.text}")

    def ws_get_course_content(self, courseid: str):
        """Run a Moodle Webservice API call.

        This function allows simple API calls via the Webservice API.

        Args:
            endpoint (str): The Moodle instance endpoint. Defaults to WS_ENDPOINT environment variable.
            courseid (str): The courseID. Defaults to COURSE_ID environment variable.
            token (str): The wstoken. Defaults to WS_TOKEN environment variable.
            fn (str): Name of Webservice API function. Defaults to core_course_get_contents.

        Returns:
            requests.Response object

        Raises:
            Exception with status code and response text.
        """
        data = {
            "courseid": courseid,
            "wstoken": self.wstoken,
            "wsfunction": "core_course_get_contents",
            "moodlewsrestformat": "json",
        }

        # HTTP request
        response = requests.post(self.wsendpoint, data)
        if response.status_code == 200:
            return response
            print("Response:")
            print(response.json())
        else:
            raise Exception(f"Request failed with status code: {response.status_code}: {response.text}")

    def ws_mark_course_as_stored(self, courseid: str):
        """Mark a course as stored in the LMSAssistant plugin.

        Args:
            courseid (str): The courseID. Defaults to COURSE_ID environment variable.

        Returns:
            requests.Response object

        Raises:
            Exception with status code and response text.
        """
        data = {
            "courseid": courseid,
            "wstoken": self.wstoken,
            "wsfunction": "local_lmsassistant_unmark_modified",
            "moodlewsrestformat": "json",
        }

        # HTTP request
        response = requests.post(self.wsendpoint, data)
        if response.status_code == 200:
            print("Mark stored:")
            print(response.json())
            return response

        else:
            raise Exception(f"Request failed with status code: {response.status_code}: {response.text}")

    def ws_mark_course_as_storing(self, courseid: str):
            """Mark a course as storing in the LMSAssistant plugin.

            Args:
                courseid (str): The courseID. Defaults to COURSE_ID environment variable.

            Returns:
                requests.Response object

            Raises:
                Exception with status code and response text.
            """
            data = {
                "courseid": courseid,
                "wstoken": self.wstoken,
                "wsfunction": "local_lmsassistant_mark_storing",
                "moodlewsrestformat": "json",
            }

            # HTTP request
            response = requests.post(self.wsendpoint, data)
            if response.status_code == 200:
                return response
                print("Response:")
                print(response.json())
            else:
                raise Exception(f"Request failed with status code: {response.status_code}: {response.text}")

    def ws_return_announcements(self, courseid: str):

        posts = pd.DataFrame(columns=['Subject', 'URL', 'Message', 'Modified'])

        fn = "mod_forum_get_forums_by_courses"

        data = {
            "courseids[0]": courseid,
            "wstoken": self.wstoken,
            "wsfunction": fn,
            "moodlewsrestformat": "json",
        }

        forumid = 0
        forums = requests.post(self.wsendpoint, data).json()
        for forum in forums:
            if (forum.get('type')=="news"):
                forumid=forum.get('id')

        if forumid==0:
            print("Unable to find forums. Returning None.")
            return None

        fn = "mod_forum_get_forum_discussions"

        data = {
            "forumid": forumid,
            "wstoken": self.wstoken,
            "wsfunction": fn,
            "moodlewsrestformat": "json",
        }

        response = requests.post(endpoint,data).json()
        discussions = response.get('discussions',[])
        for discussion in discussions:
            fn = "mod_forum_get_discussion_posts"
            data = {
                "discussionid": discussion.get('discussion'),
                "wstoken": self.wstoken,
                "wsfunction": fn,
                "moodlewsrestformat": "json",
            }
            discussion_posts = requests.post(endpoint,data).json().get('posts',[])
            for post in discussion_posts:
                subject=post['subject']
                message=post['message']
                url=post['urls']['view']
                modified=post['timemodified']
                posts.loc[len(posts)]=[subject,url,message, modified]

        return(posts)

    def ws_create_file_list(self, response: requests.Response):
        """Create file list from Moodle Webservice core_course_get_contents response.

        This function create a file list from Moodle Webservice API response object.

        Args:
            response (requests.Response): requests.Response object based on ws_get_course_content(courseid)

        Returns:
            pandas DataFrame
        """
        # Initialize dataframe for metadata
        file_data = pd.DataFrame(columns=['Filename','User URL','Download URL', 'Modified'])

        response=response.json()
        for section in response:
            for module in section.get('modules', []):
                if module.get('modname','')=='resource' or module.get('modname','')=='folder':
                    module_name = module.get('name','')
                    module_url = module.get('url','')
                    for content in module.get('contents', []):
                        item_name = module_name+"->"+content.get('filename','')
                        file_data.loc[len(file_data)] = [item_name,module_url,content.get('fileurl', '') + "&token=" + self.wstoken, content.get('timemodified', '')]
        return file_data

    def ws_files_todisk(self, df: pd.DataFrame, dirfiles: str):
        """Save Moodle files to disk.

        This function downloads files based on a file list and saves them to disk.

        Args:
            df (pandas.DataFrame): pandas.DataFrame object that contains a list of files to be saved
            save_location (str): The directory to save the files. Defaults to WS_STORAGE environment variable.

        Returns:
            boolean: Whether the save was successful or not.
        """
        save_dir = self.wsstorage + "/" + dirfiles
        if save_dir is None:
            print("No save folder determined. Exiting.")
            return False

        if os.path.isdir(self.wsstorage) == False:
            os.mkdir(self.wsstorage)

        if os.path.isdir(save_dir) == False:
            os.mkdir(save_dir)

        for index, row in df.iterrows():
            print("Download file :" + str(['Download URL']))
            # Download file from URL
            urlretrieve(row['Download URL'], save_dir + "/" + row['Filename'])

        return True

    def read_filenames_from_directory(self, material_directory: str):
        filenames = []
        for root, dirs, files in os.walk(material_directory):
            for name in files:
                # Exclude dot-files
                if name[0] != '.':
                    filenames.append(os.path.join(root, name))
        return filenames

    def create_material_headings_from_filenames(self, filenames, material_directory):
        # Make headings pretty based on file names
        # '_' to ' ', remove file suffixes, title case, "/" to ": "
        material_headings = [filename[len(material_directory):] for filename in filenames]
        def pretty_headings(heading):
            heading = heading.replace('_', ' ')
            heading = heading.split('.')[0]
            heading = heading.title()
            heading = heading.replace('/', ': ')
            return heading
        material_headings = [pretty_headings(heading) for heading in material_headings]
        return material_headings

    def convert_files_totext(self, filenames):
        # Extract text from the files
        # Supported file formats: https://textract.readthedocs.io/en/stable/ + MarkDown
        texts = []
        for filename in filenames:
            # Exctract file type
            filetype = filename.split('.')[-1]
            print("Converting to text: " + filename)
            if filetype != "md":
                text = textract.process(filename)
                text = text.decode("utf-8")
            else:
                with open(filename) as f:
                    text=f.read()
                    f.close()

            texts.append(text)
        return texts

    def create_chunk_dataframe(self, material_headings, texts, max_size=500):
        df = pd.DataFrame({'Heading': material_headings, 'Text': texts})

        # Initialisation du tokenizer et du modèle
        model_name = "facebook/bart-large-cnn"
        bart_tokenizer = BartTokenizer.from_pretrained(model_name)
        bart_model = BartForConditionalGeneration.from_pretrained(model_name)


        # Initialisation du text_splitter avec une grande taille pour commencer
        # La logique de contrôle réduira la taille si nécessaire
        text_splitter = CharacterTextSplitter(
            separator="\n\n",
            chunk_size=max_size,  # Taille initiale grande, ajustée plus bas
            chunk_overlap=0,
            length_function=len,
            is_separator_regex=False,
        )

        def split_text_adjusting_for_heading(row):
           heading = "Source: " + row['Heading'] + '\n'
           initial_text = row['Text']
           available_text_size = max_size - len(heading)
           text_splitter.chunk_size = max(available_text_size, 100)
           text_chunks = text_splitter.split_text(initial_text)

           # Vérifiez et résumez les chunks trop longs
           adjusted_chunks = []
           for chunk in text_chunks:
               if len(chunk) > max_size:
                   inputs = bart_tokenizer.encode("summarize: " + chunk, return_tensors="pt", max_length=available_text_size, truncation=True)
                   summary_ids = bart_model.generate(inputs, max_length=available_text_size, length_penalty=2.0, num_beams=4, early_stopping=True)
                   chunk = bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
               adjusted_chunks.append(heading + chunk)

           return adjusted_chunks


        # Appliquer la fonction de segmentation ajustée à chaque ligne du DataFrame
        df['Text_Splitted_w_Headings'] = df.apply(split_text_adjusting_for_heading, axis=1)
        return df

    def create_vector_store(self, df, metadatas=False):
        master_chunk = []
        master_metadata=[]
        for i, row in df.iterrows():
            master_chunk += row['Text_Splitted_w_Headings']
            if metadatas:
                for text_in_row in row['Text_Splitted_w_Headings']:
                    master_metadata.append(row[['Heading','Modified']].to_dict())
        # Create vector store
        embeddings = HuggingFaceInstructEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        return FAISS.from_texts(texts=master_chunk, embedding=embeddings,metadatas=master_metadata if metadatas else None)

    def store(self):
        print("Starting LMSAssistant importation." )
        # Retrieve courses needing to be stored.
        courses = self.ws_get_courses_to_vectorstore()

        # For each course, retrieve the course contents and store them to disk.
        for course in courses:

            # Course ID.
            courseid = course.get('id')

            # Mark Moodle Course as storing.
            self.ws_mark_course_as_storing(courseid)

            # Retrieve course contents.
            resp=self.ws_get_course_content(courseid)

            # Create file list.
            df=self.ws_create_file_list(resp)

            # Directory to save files.
            dirfiles = "course_" + str(courseid)

            # Save files to disk in the directory.
            self.ws_files_todisk(df, dirfiles)
            #posts=ws_return_announcements()

            # Keep track of the files that were downloaded.
            filenames = []
            for file in df['Filename']:
                file = self.wsstorage + "/" + dirfiles + "/" + file
                print(file)
                filenames.append(file)

            # Convert files to text.
            texts = self.convert_files_totext(filenames)

            # Create material headings.
            material_headings = df['Filename'] + ", " + df['User URL']
            material_headings = material_headings.tolist()

            # Create chunks of headings dataframe.
            chunk_df = self.create_chunk_dataframe(material_headings, texts)
            chunk_df['Modified'] = df['Modified']

            # print(posts)
            #if posts is not None:
            #    post_headings = "Announcements->" + posts['Subject'] + ", " + posts['URL']
            #    post_chunk_df = create_chunck_dataframe(post_headings,posts['Message'])
            #    post_chunk_df['Modified'] = posts['Modified']
            #    chunk_df = pd.concat([chunk_df,post_chunk_df],ignore_index=True)

            # Create vector store faiss.
            vector_store = self.create_vector_store(chunk_df, metadatas=True)

            # File to save vector store in vector_stores/course_ + courseid.
            vector_store_dir = "vector_stores/" + dirfiles
            if os.path.isdir(vector_store_dir)==False:
                os.mkdir(vector_store_dir)
            vector_store.save_local(vector_store_dir)

            # Mark Moodle Course as stored.
            self.ws_mark_course_as_stored(courseid)

            # Remove the course directory.
            os.system("rm -rf " + dirfiles)
