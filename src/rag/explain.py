import os
from typing import List

from huggingface_hub import InferenceClient


class RecommendationExplainer:
    def __init__(self, model_name: str = 'Qwen/Qwen2.5-7B-Instruct',
                 timeout: float = 8.0):
        self.api_key = os.getenv('HF_TOKEN', '')
        self.model_name = model_name
        self.client = None
        if self.api_key:
            self.client = InferenceClient(
                model=self.model_name, token=self.api_key, timeout=timeout,
            )

    def generate_explanation(self, user_history_titles: List[str],
                              rec_title: str, rec_genres: str) -> str:

        if self.client is None:
            return self._fallback_explanation(user_history_titles, rec_title, rec_genres)

        history_str = ', '.join(user_history_titles[-3:]) if user_history_titles else 'popular picks'
        prompt = (
            f"You are a helpful movie recommendation assistant.\n"
            f"The user recently watched and liked: {history_str}.\n"
            f"Explain in 1-2 short, engaging sentences why they would enjoy "
            f"watching '{rec_title}' (Genres: {rec_genres}).\n"
            f"Keep it concise and direct. Do not use quotes or introductory fluff."
        )

        try:
            response = self.client.chat.completions.create(
                messages=[{'role': 'user', 'content': prompt}],
                max_tokens=80,
                temperature=0.6,
            )
            explanation = response.choices[0].message.content.strip()
            return explanation if explanation else self._fallback_explanation(
                user_history_titles, rec_title, rec_genres)
        except Exception as e:
            print(f'[RAG explain] API call failed, using fallback: {e}')
            return self._fallback_explanation(user_history_titles, rec_title, rec_genres)

    @staticmethod
    def _fallback_explanation(history: List[str], rec_title: str, rec_genres: str) -> str:
        genres_readable = rec_genres.replace('|', ', ')
        if history:
            last_movie = history[-1]
            return f"Recommended because you watched {last_movie} and enjoy {genres_readable} films."
        return f"Recommended as a top-rated choice matching popular interest in {genres_readable}."


if __name__ == '__main__':
    explainer = RecommendationExplainer()
    if explainer.client is None:
        print('No HF_TOKEN set — testing fallback path only.\n')
    else:
        print(f'HF_TOKEN found — will attempt a live API call to {explainer.model_name}.\n')

    result = explainer.generate_explanation(
        user_history_titles=['Star Wars: Episode IV', 'Terminator 2'],
        rec_title='The Matrix',
        rec_genres='Action|Sci-Fi',
    )
    print('Generated explanation:', result)