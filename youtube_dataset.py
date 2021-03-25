import pandas as pd
from torch.utils.data import Dataset

class YoutubeDataset(Dataset):
    def __init__(self,videos_data, channels_data):
        super().__init__()
        videos = pd.read_csv(videos_data)
        channels = pd.read_csv(channels_data)
        videos['title_length'] = videos.apply(lambda row: self._get_title_length(row['title']), axis=1)
        videos['number_of_tags'] = videos.apply(lambda row: self._get_number_of_tags(row['tags']), axis=1)
        videos['title_contains_all_caps_word'] = videos.apply(lambda row: self._contains_all_caps_word(row['title']), axis=1)
        channels = channels.rename(columns={'title':'channel_title', 'category_name': 'channel_category_name', 'category_id': 'channel_category_id'})
        self.data = videos.merge(channels, on='channel_title', how='left')
        self.data = self.data.dropna(subset=['followers'])
        self.data = self.data.reset_index(drop=True)
        self.numerical_variables = ['followers', 'videos', 'title_length', 'number_of_tags']
    def _get_title_length(self, title):
        return len(title)

    def _get_number_of_tags(self, string_of_tags):
        tags = string_of_tags.split('|')
        tags_no_empty_strings = []
        for tag in tags:
            if tag != '':
                tags_no_empty_strings.append(tag)
        return len(tags_no_empty_strings)

    def _contains_all_caps_word(self, title):
        words = title.split(' ')
        words_in_capital = [word for word in words if word.isupper()]
        return 1 if len(words_in_capital) >=3 else 0

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return (
            self.data.loc[idx, self.numerical_variables].values
        )
