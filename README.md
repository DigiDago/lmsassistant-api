# API Moodle files stored in vector storage

This is a repository for the API storing the files of the courses of Moodle platform in a vector storage.
It enables to request accurate information about the files of the courses in the platform thankfully to the OpenAI API.

## Installation

Clone the repository and install the requirements:

```
pip3 install -r requirements.txt
```

## Usage

Set the environnement variables (.env file) :

```
LLM_PROVIDER=<LLM Provider to use ex: openai>
API_KEY=<API key enabling the access to this API>
MODEL_NAME=<Model name to use ex: gpt-3.5-turbo>
DOC_LANGUAGE=<Language of the documents ex: en>
OPENAI_API_KEY=<OpenAI API key>
MAX_PROMPT_TOKENS=<Max tokens to use in prompt ex: 2048>
MAX_CONCURRENCY=<Max concurrency to use ex: 10>
MAX_QUEUE=<Max queue to use ex: 10>
MODEL_CACHE=<Cache directory to use ex: cache> 
WS_TOKEN=<Moodle Web Service token>
WS_ENDPOINT=<Moodle Web Service endpoint ex: https://moodle.example.com/webservice/rest/server.php>
WS_STORAGE=<Downloading files temporary storage directory ex: moodle_file>
```

Do not forget to create token in Moodle and give it the right permissions to access to local_lmsassistant function and core_course_get_contents function.

Launch the application:

```
uvicorn main:app --host 0.0.0.0 --port 8080
```


Maintainer
============
Lmsassistant is developed by Pimenko Team.
[https://pimenko.com/](https://pimenko.com/)

Support
===================
This project is being developed with the support of Lab e-nov
[https://www.ifp-school.com/recherche-et-innovation/lab-enov](https://www.ifp-school.com/recherche-et-innovation/lab-enov)


Any Problems, questions, suggestions
===================
If you have a problem with this theme, suggestions for improvement, drop an email via :
- Github
- Your webiste: [https://pimenko.com/](https://pimenko.com/contact-mooc-elearning/)
