import json
import numpy as np
import crowdai

api_key = 'bc2f7cddee3eb0c65f0737ec22bf1c48'#'a96a541f34d8c0d688871eab01ba8057'
challenge = crowdai.Challenge('crowdAIMappingChallenge', api_key)
result = challenge.submit('newpredictions.json')
print(result)
