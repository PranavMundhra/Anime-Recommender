import unittest
from unittest.mock import patch
from anime_recommender.conversation import initialize, reset, begin_turn, stream_turn, retry_turn, build_messages, recover_interrupted_turn, wants_recommendations
from anime_recommender.services.llm import ProviderClient

class Fake:
    def __init__(self, fail=False): self.fail=fail; self.calls=0
    def stream(self, messages):
        self.calls += 1
        yield 'Hello '
        if self.fail: raise TimeoutError('secret')
        yield 'there.'

class ConversationTests(unittest.TestCase):
    def setUp(self): self.state={}; initialize(self.state)
    def test_message_visible_before_stream_and_sent_once(self):
        begin_turn(self.state, 'I enjoy comedy')
        self.assertEqual(self.state['conversation_history'][0]['content'], 'I enjoy comedy')
        self.assertEqual(sum(m['content']=='I enjoy comedy' for m in build_messages(self.state)),1)
        stream=stream_turn(self.state, Fake())
        self.assertEqual(next(stream),'Hello ')
        self.assertEqual(len(self.state['conversation_history']),1)
        list(stream)
        self.assertEqual(self.state['completed_turns'],1)
        self.assertEqual(len(self.state['conversation_history']),2)
        with self.assertRaises(RuntimeError): list(stream_turn(self.state,Fake()))
    def test_failure_retry_does_not_duplicate_or_consume_turn(self):
        begin_turn(self.state,'Comedy please')
        with self.assertRaises(TimeoutError): list(stream_turn(self.state,Fake(True)))
        self.assertEqual(self.state['completed_turns'],0)
        self.assertEqual(len(self.state['conversation_history']),1)
        self.assertEqual(self.state['pending_turn']['status'],'failed')
        self.assertNotIn('secret',self.state['pending_turn']['error'])
        retry_turn(self.state)
        list(stream_turn(self.state,Fake()))
        self.assertEqual(len(self.state['conversation_history']),2)
        self.assertEqual(self.state['completed_turns'],1)
    def test_overlap_and_rerun_recovery(self):
        begin_turn(self.state,'Hello')
        with self.assertRaises(RuntimeError): begin_turn(self.state,'Another')
        self.state['pending_turn']['status']='processing'
        recover_interrupted_turn(self.state)
        self.assertEqual(self.state['pending_turn']['status'],'failed')
    def test_cancelled_generator_is_incomplete(self):
        begin_turn(self.state,'Hello')
        stream=stream_turn(self.state,Fake()); next(stream); stream.close()
        self.assertEqual(self.state['pending_turn']['status'],'failed')
        self.assertEqual(self.state['completed_turns'],0)
    def test_reset_preserves_unrelated_state(self):
        self.state['provider_select']='mistral'; self.state['other']=42
        begin_turn(self.state,'Hello'); reset(self.state)
        self.assertEqual(self.state['provider_select'],'mistral')
        self.assertEqual(self.state['other'],42)
        self.assertEqual(self.state['conversation_history'],[])
    def test_installed_gemini_constructor_offline(self):
        with patch('anime_recommender.services.llm.require_api_key',return_value='dummy-offline-key'):
            client=ProviderClient('gemini')
        self.assertEqual(client.client.max_retries,1)
        self.assertGreater(client.client.timeout,0)
        self.assertFalse(client.client.include_thoughts)
    def test_empathy_prompt_keeps_context_without_a_collection_checklist(self):
        begin_turn(self.state,'Hello')
        first=build_messages(self.state)[0]['content']
        self.assertIn('acknowledge what they actually said', first)
        self.assertIn('Never require genre knowledge', first)
        with self.assertRaises(TimeoutError): list(stream_turn(self.state,Fake(True)))
        self.assertEqual(build_messages(self.state)[0]['content'],first)
        retry_turn(self.state); list(stream_turn(self.state,Fake()))
        begin_turn(self.state,'Not sure')
        second=build_messages(self.state)
        self.assertNotIn('Collection phase', second[0]['content'])
        self.assertEqual(sum(m['content']=='Not sure' for m in second),1)
        fresh={}; initialize(fresh); begin_turn(fresh,'I want relaxing comedy and no horror')
        self.assertNotIn('invite profile review now',build_messages(fresh)[0]['content'])
    def test_suggestion_requests_do_not_confuse_feelings_or_questions(self):
        for message in ('Recommend something gentle', 'Can you suggest an anime?', 'What should I watch?', 'Give me alternatives'):
            self.assertTrue(wants_recommendations(message), message)
        for message in ("I'm feeling lonely", "Don't recommend anything yet", 'Why did you recommend this?', 'These suggestions are too sad'):
            self.assertFalse(wants_recommendations(message), message)
        history = [{'role': 'assistant', 'content': 'Would you like me to suggest a gentle show?'},
                   {'role': 'user', 'content': 'Yes please'},
                   {'role': 'assistant', 'content': 'I will look for warm friendships.'}]
        self.assertTrue(wants_recommendations('Yes please', history))
        self.assertFalse(wants_recommendations('Yes please', []))
        offer = [{'role': 'user', 'content': 'Long day'},
                 {'role': 'assistant', 'content': 'Want me to find a few shows that might help?'},
                 {'role': 'user', 'content': 'x'}]
        for reply in ('Sounds good', "I'd like that", 'go for it!', 'Yes, thanks'):
            self.assertTrue(wants_recommendations(reply, offer), reply)
        chat = [{'role': 'assistant', 'content': 'How was your day?'}, {'role': 'user', 'content': 'x'}]
        self.assertFalse(wants_recommendations('Sounds good', chat))
        for reply in ('Yeah, that would be really nice actually', 'Definitely!', 'oh yes please, thank you'):
            self.assertTrue(wants_recommendations(reply, offer), reply)
        for reply in ('No thanks, I just want to talk', 'Yeah maybe later', 'Not yet', "I'd rather keep chatting"):
            self.assertFalse(wants_recommendations(reply, offer), reply)
    def test_search_marker_is_hidden_even_when_split_and_requests_search(self):
        class Marker:
            def stream(self, messages):
                yield 'I will look for something warm and calm. [[FIND'
                yield '_SHO'
                yield 'WS]]'
        begin_turn(self.state, 'Yeah, that would be lovely')
        shown = ''.join(stream_turn(self.state, Marker()))
        self.assertEqual(shown.strip(), 'I will look for something warm and calm.')
        self.assertEqual(self.state['conversation_history'][-1]['content'], 'I will look for something warm and calm.')
        self.assertTrue(self.state['search_requested'])
    def test_search_needs_direct_request_or_answer_to_an_offer(self):
        from anime_recommender.conversation import should_search
        def state(*messages, marker):
            roles = ('user', 'assistant') * len(messages)
            return {'conversation_history': [{'role': r, 'content': c} for r, c in zip(roles, messages)],
                    'search_requested': marker}
        # The model offered and marked in the same reply: ignored, the user must answer first.
        self.assertFalse(should_search(state('I want something calm', 'Want me to find a few calm shows?', marker=True)))
        # A loosely worded yes to the previous offer is honoured via the marker.
        self.assertTrue(should_search(state('Long day', 'Want me to find a few calm shows?',
                                            'honestly that would really help right now', 'On it.', marker=True)))
        # Without an offer, the marker alone never starts a search.
        self.assertFalse(should_search(state('ugh', 'Tell me more?', 'I just want to feel better', 'Okay.', marker=True)))
        # A decline after an offer never searches, whatever the model marked.
        self.assertFalse(should_search(state('Long day', 'Want me to find a few shows?', 'not yet, let me vent', 'Sure.', marker=True)))
        # Asking directly always works.
        self.assertTrue(should_search(state('Can you recommend something cozy?', 'Sure!', marker=False)))
    def test_text_resembling_marker_start_is_not_lost_and_no_search(self):
        class Brackets:
            def stream(self, messages):
                yield 'Tell me more [['
                yield 'about your day]]'
        begin_turn(self.state, 'Hi')
        self.assertEqual(''.join(stream_turn(self.state, Brackets())), 'Tell me more [[about your day]]')
        self.assertFalse(self.state['search_requested'])
    def test_companion_is_told_never_to_name_titles(self):
        begin_turn(self.state, 'Hi')
        prompt = build_messages(self.state)[0]['content']
        self.assertIn('Never name, list or describe specific anime titles', prompt)
        self.assertIn('[[FIND_SHOWS]]', prompt)
    def test_installed_mistral_constructor_offline(self):
        with patch('anime_recommender.services.llm.require_api_key',return_value='dummy-offline-key'):
            client=ProviderClient('mistral')
        self.assertEqual(str(client.client.base_url),'https://api.mistral.ai/v1/')
        self.assertEqual(client.client.max_retries,1)
        self.assertGreater(client.client.timeout,0)
        client.client.close()

if __name__=='__main__': unittest.main()
