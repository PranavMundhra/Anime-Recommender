import unittest
from pydantic import ValidationError
from anime_recommender.profile import UserProfile, extract_profile, local_profile, readiness
from anime_recommender.services.llm import Response, text_content

class Fake:
    def __init__(self,text): self.text=text
    def invoke(self,messages): return Response(self.text)

class ProfileTests(unittest.TestCase):
    def test_schema_rejects_invalid_fields(self):
        for fields in ({'confidence':1.1},{'mood_level':'happy'},{'max_episodes':-1}, {'min_score':11}, {'genres':'action'}, {'min_episodes':20,'max_episodes':10}):
            with self.assertRaises(ValidationError): UserProfile(**fields)
    def test_normalizes_labels(self):
        self.assertEqual(UserProfile(genres=['slice-of-life','action','Action','science fiction']).genres,['Slice of Life','Action','Sci-Fi'])
    def test_detailed_first_message_ready_without_network(self):
        profile=local_profile([{'role':'user','content':'I want relaxing comedy, no horror, under 13 episodes. I have watched Naruto.'}])
        self.assertEqual(readiness(profile),[])
        self.assertEqual(profile.excluded_genres,['Horror'])
        self.assertEqual(profile.max_episodes,12)
        self.assertIn('Naruto',profile.watched_titles)
    def test_vague_response_needs_intent(self):
        self.assertTrue(readiness(local_profile([{'role':'user','content':'Hi'}])))
        for message in ('I do not know what to watch', 'I am not sure, recommend anything',
                        'I would like to watch anime', 'Please suggest a show to watch'):
            self.assertTrue(readiness(local_profile([{'role':'user','content':message}])),message)
    def test_meaningful_experience_without_genre_is_ready(self):
        for message in ('I want something uplifting after a long day',
                        'Not sure, but something relaxing would help me unwind',
                        'I would enjoy a slow philosophical story'):
            profile=local_profile([{'role':'user','content':message}])
            self.assertEqual(profile.genres,[])
            self.assertEqual(readiness(profile),[],message)
    def test_invalid_json_or_schema_returns_honest_draft(self):
        history=[{'role':'user','content':'No romance please, comedy under 20 episodes'}]
        for output in ('bad json','{"confidence":9}','{"genres":"Action"}'):
            profile=extract_profile(history,client=Fake(output))
            self.assertTrue(profile.is_fallback)
            self.assertIn('Romance',profile.excluded_genres)
            self.assertEqual(profile.confidence,0)
            self.assertTrue(profile.notices)
    def test_self_reported_feeling_is_usable_without_genres(self):
        for message in ("I'm feeling lonely", 'I feel overwhelmed', "I've been feeling really tired"):
            profile = local_profile([{'role': 'user', 'content': message}])
            self.assertTrue(profile.emotions, message)
            self.assertFalse(profile.genres)
            self.assertFalse(readiness(profile))
            self.assertIn('gentle', profile.recommendation_focus)
    def test_desired_experience_overrides_distress_for_matching(self):
        history = [{'role': 'user', 'content': "I'm sad and lonely"},
                   {'role': 'assistant', 'content': 'Would comfort or catharsis help tonight?'},
                   {'role': 'user', 'content': 'I want to laugh and be distracted'}]
        profile = local_profile(history)
        self.assertIn('funny', profile.recommendation_focus)
        self.assertNotIn('loss', profile.recommendation_focus)
        history.append({'role': 'user', 'content': 'Actually I want to cry and process my feelings'})
        self.assertIn('reflective', local_profile(history).recommendation_focus)
    def test_story_emotion_and_negated_feelings_are_not_self_reports(self):
        for message in ('I want a sad anime', "I'm not sad", 'A story about someone who feels lonely'):
            self.assertFalse(local_profile([{'role': 'user', 'content': message}]).emotions, message)
    def test_model_cannot_drop_user_exclusions(self):
        profile=extract_profile([{'role':'user','content':'I enjoy action. No horror. Under 13 episodes.'}],client=Fake('{"genres":["Horror","Action"],"max_episodes":50}'))
        self.assertEqual(profile.genres,['Action'])
        self.assertIn('Horror',profile.excluded_genres)
        self.assertEqual(profile.max_episodes,12)
    def test_only_text_blocks_are_displayed(self):
        self.assertEqual(text_content([{'type':'reasoning','text':'hidden'},{'type':'text','text':'hello'}]),'hello')
    def test_installed_gemini_mixed_thought_content_preserves_visible_text(self):
        from langchain_google_genai.chat_models import _parse_response_candidate
        from google.ai.generativelanguage_v1beta.types import Candidate, Content, Part
        candidate = Candidate(content=Content(parts=[Part(text='private reasoning',thought=True),Part(text='Visible answer')]))
        message = _parse_response_candidate(candidate,streaming=True)
        self.assertEqual(text_content(message),'Visible answer')

if __name__=='__main__': unittest.main()
