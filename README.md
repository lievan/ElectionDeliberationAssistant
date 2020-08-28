# ElectionDeliberationAssistant
This program takes a text snippet as input and returns sentences from Wikipedia relavant to Trump and Biden in relation to the text snippet using GloVe and argBERT. 

# Downloads
```
pip install requirements.txt

```
Download stanford glove vectors
```
!wget http://nlp.stanford.edu/data/glove.6B.zip

```
```
!unzip glove.6B.zip

```
Download argBERT-standard from 
```
https://drive.google.com/drive/folders/1wz7BB7FcaS1V6mTBZM8XPpym-t89Ycx7?usp=sharing
```


# Model initialization

**SemanticSearcher model**

We use SemanticSearcher to return the top 200 closest arguments in semantic space to the query

```
similarity_model = ElectionAssistant.SemanticSearcher('glove.6B.300d.txt')
```

**argBERT model**

We use argBERT to select the top_n most relevant responses to the query out of the the 200 arguments that SemanticSearcher gives us. argBERT requires a GPU.

```
argBERT_model = ElectionAssistant.argBERT('path_to/argBERT-standard', 'cuda')
```
**Corpus**

Initialize lists of arguments we select response reccomendations from

```
trump_list, biden_list = ElectionAssistant.get_arg_list('path_to/trump.txt', 'path_to/biden.txt')
```

# Get response reccomendations

```
query = 'Foreign relations with China'
trump, biden = ElectionAssistant.get_responses(query, trump_list, biden_list, argBERT_model, similarity_model, top_n=5)
```

```
print('For Trump, Wikipedia mentions that: ')
print('                                    ')
for arg in trump:
  print(arg[0])

print('                                    ')
print('For Biden, Wikipedia mentions that: ')
print('                                    ')

for arg in biden:
  print(arg[0])
```
Output:
```
For Trump, Wikipedia mentions that: 
                                    
On 28 December 2017, U.S. President Donald Trump accused the Chinese government of "allowing oil to go into North Korea."[REF] 
Early in Trump's presidency, the world's largest financial newspaper, Nikkei Asian Review, had reported on February 1, that Trump had labelled China and Japan as currency manipulators [REF]. 
Concerned that China was acquiring advanced American technology, the Trump administration prepared to announce plans by June 30, 2018 to restrict Chinese investment in American technology companies and set technology export controls for China [REF]. 
In October 2017 The Wall Street Journal reported that Wynn, who has financial interests in China, lobbied President Trump on behalf of the Chinese government to return a Chinese dissident, Guo Wengui, to China [REF]. 
In 2017, Schumer wrote to President Trump advocating for a block on China that would prevent the latter country from purchasing more American companies to increase pressure on Beijing to help rein in the nuclear missile program of North Korea [REF]. 
                                    
For Biden, Wikipedia mentions that: 
                                    
On February 9, 2012, Liebman met with Vice President of the United States Joe Biden at the White House to discuss human rights in China [REF]. 
On 4 December 2013, American vice president Joe Biden discussed the issue at length with Chinese president Xi Jinping [REF]. 
(AAP via News Limited) Vice President of the United States Joe Biden criticizes the People's Republic of China for a recent crackdown of foreign journalists in the country. 
A large United Nations exhibit includes this quotation from Vice President Joe Biden: "Just as only Nixon could go to China, only Helms could fix the U.N.", a reference to Helms' reform initiatives regarding the world body [REF]. 
4 December  During a meeting in Beijing, Vice President of the United States Joe Biden warns Chinese President Xi Jinping not to establish another air defense information zone over disputed waters in the South China Sea like the one the People's Republic of China unilaterally declared over the East China Sea in November [REF]. 
```
