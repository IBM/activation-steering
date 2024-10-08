import dataclasses
import typing
from typing import List
import warnings
import time

import torch
from transformers import PretrainedConfig, PreTrainedModel
from rich.progress import Progress, TimeElapsedColumn, TimeRemainingColumn, BarColumn, TextColumn


@dataclasses.dataclass
class LayerControlParams:
    control: torch.Tensor | None = None
    operator: typing.Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = (
        lambda current, control: current + control
    )
    """
    A dataclass for layer control parameters.

    Attributes:
        control: Optional tensor for control.
        operator: Callable that defines how to apply the control to the current state.
    """

    @classmethod
    def default(cls):
        """
        Return a default instance of LayerControlParams.

        Returns:
            A LayerControlParams instance with default values.
        """
        return cls()


@dataclasses.dataclass
class ContrastivePair:
    """
    A dataclass representing a pair of contrasting strings.

    Attributes:
        positive: The positive string in the pair.
        negative: The negative string in the pair.
    """
    positive: str
    negative: str


def custom_progress(iterable, description):
    """
    Create a custom progress bar for iterating over items.

    Args:
        iterable: The iterable to process.
        description: A string describing the progress bar.

    Yields:
        Items from the iterable.
    """
    items = list(iterable)  # Convert iterable to a list
    total = len(items)
    
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        TextColumn(""),
        expand=True
    ) as progress:
        task = progress.add_task(description, total=total)
        
        for i, item in enumerate(items):
            yield item
            progress.update(task, completed=i+1)
            
        # Ensure 100% completion
        progress.update(task, completed=total)
        time.sleep(0.1)  # Small delay to ensure the final update is displayed


def return_default_suffixes():
    """
    Return a list of default suffixes used in the CAIS representation engineering paper.

    Returns:
        A list of string suffixes.
    """
    return [ "", "That game", "I can see", "Hmm, this", "I can relate to", "Who is", "I understand the", "Ugh,", "What the hell was", "Hey, did anyone", "Although", "Thank you for choosing", "What are you", "Oh w", "How dare you open", "It was my pleasure", "I'm hon", "I appreciate that you", "Are you k", "Whoever left this", "It's always", "Ew,", "Hey, I l", "Hello? Is someone", "I understand that", "That poem", "Aww, poor", "Hey, it", "Alright, who", "I didn't", "Well, life", "The document", "Oh no, this", "I'm concerned", "Hello, this is", "This art", "Hmm, this drink", "Hi there!", "It seems", "Is", "Good", "I can't", "Ex", "Who are", "I can see that", "Wow,", "Today is a", "Hey friend", "Sometimes friends", "Oh, this old", "The weather outside", "This place is sur", "I appreciate your input", "Thank you for the", "Look at", "I'm disappoint", "To my", "How dare you", "That's an", "This piece of art", "Eww", "This park is", "This is incredible", "Oh no, someone", "Exc", "Well, it'", "I warned", "Hey, I understand", "Hey, I saw", "How dare you go", "What the he", "Hey", "It's", "Hello? Hello?", "It", "Oh no!", "This is the perfect", "Good morning,", "Oh no, there", "It's so", "Yeah", "Uh,", "Hello everyone", "Who turned off", "The weather", "Who'", "Hey, this", "Wait,", "Eww, gross", "Excuse", "It seems like you", "Thank you so", "What happened?", "Oh my g", "I am deeply sad", "I war", "Okay, let'", "Hey, that", "That was a beautiful", "Oh no! That", "What happened", "Hey there", "The artist'", "What?!", "Hey, it'", "I am disappoint", "It seems like", "Oh no! The", "This park is a", "If you", "Yes! I did", "It sounds", "What", "Who is it", "Hmm, that", "That's strange", "Yeah, that was", "That's interesting", "This park", "What the hell", "Who is that", "I feel like my", "Oh well", "What the hell is", "Hello? Hello", "To my dearest", "Bless you!\"", "Thank you for", "Oh, looks like", "Can you please", "This place is", "Eww, what", "Bless you", "Is everything", "Hey, I just", "Whoever left these", "Well, that'", "I feel", "Hey, do you", "It's sad", "Oh no, it", "Hey, that'", "Oh my god,", "Thank you,", "Hello little one,", "I apolog", "Hey team, I", "How dare you read", "Who is this and", "Whoever left", "Hi there! W", "A", "If you have", "I was", "U", "Bless", "Well, this", "Oh, I'", "It's a", "Eww,", "Is everything okay?", "Oh, I", "Hello, can you", "Al", "That was a great", "What are", "I understand that not", "Oh no, not", "Who is it?\"", "Hey, can we", "Whoever is taking", "I would love to", "Hey, I noticed", "Hey, could", "I understand that there", "Hello?", "D", "Oh man, I", "Thank you so much", "Oh no, my", "Dear [Name", "Uh", "I remember", "Hey, who", "Well, it", "Are you", "I understand that it", "Hey, is", "I would", "Who is this", "Excuse me", "Alright", "I am thrilled", "Sometimes friends have", "Who the", "It's interesting", "I would love", "E", "Hello? Is anyone", "Well, this is", "This place", "Well,", "I warned you", "Hey, watch where", "Oh my", "That'", "Sometimes friends have different", "I understand that everyone", "What?", "What do these notes", "I can relate", "I'm not", "I understand", "To my dear", "Guys", "Well", "Hey, I appreciate", "Wow, what", "Dear", "That melody", "Who the hell", "Today is", "Hello little", "Wow, look", "That's great", "Love is never wrong", "I'm having", "Whoa, did", "Ugh", "Can you please provide", "I miss you,", "I feel uncom", "I know", "Ugh, this", "Hey, watch", "Oh great, a", "I didn", "Okay", "That game of char", "Oh", "I appreciate", "Who's there", "I am so", "Oh great, someone", "Hey, could you", "I remember wondering", "Wait, what?", "What do", "Hello? Can", "Hey there,", "That game of", "This is incred", "Oh my gosh", "Oh great, f", "I appreciate your", "It sounds like", "What the heck", "Okay, I understand", "Ew", "I understand that this", "Uh, hi", "Hi everyone!", "What the hell?", "Thank you for your", "Oh no, the", "Wow, I", "Who turned", "Dear [", "Whoever", "This is a", "Whoa, he", "What in the world", "Although the physical", "Hello, who is", "That's amaz", "Hey, I know", "Okay, that", "Hi everyone", "Hey, is everything", "I understand your fr", "Oh no, poor", "Oh, look", "Good morning", "Ew, gross", "Oh no, did", "Look at the family", "Hey team", "Yes!", "Hey, can I", "Okay, that'", "It's great", "Love is", "Hey, what", "Good morning, world", "Who is it?", "That poem really reson", "I", "That's", "I understand the task", "Gu", "Hello? Who'", "This postcard is", "Whoa,", "Oh, that", "I understand that I", "Whoever is", "Hello? Who is", "I'm really", "Wow, this", "Can", "This artwork really", "This is a shame", "I miss you too", "Who are you?", "Today is a difficult", "Hey, just", "Are you okay", "I am", "Hi,", "Wow, that", "Hey there! Can", "Okay, stay", "Oh great, just", "Yeah,", "Hello? Can you", "Oh, looks", "Thank you for sharing", "I'm glad", "Hey, is that", "Hmm", "It was my", "It sounds like you", "Wow, your", "I was promised certain", "That was such a", "Thank", "Excuse you", "That was", "Hey team,", "I feel un", "It was", "What'", "Hey friend, I", "How", "Saying goodbye", "That", "It's heart", "How dare", "Oh,", "Hello, may", "What's this", "Thank you for recogn", "Aww, that", "Oh, I remember", "Hmm, that'", "I miss", "I know this", "Wait", "Is everything okay", "Who is that person", "Wow, you", "Oh great", "I'm sad", "Wow, the", "I am very disappoint", "Who turned off the", "I understand that things", "I'm very", "Hi", "That's very", "Okay, I", "Oh no,", "Wow, there", "What's wrong", "I apologize for", "Hey, I", "Can I help you", "Oh, I didn", "Alright,", "Oh wow,", "Oh my goodness", "I know this event", "What in the", "Saying", "Yeah, that", "Guys, I", "Hey, this v", "This post", "Are", "Hey, can", "Hello? Is", "I can only imagine", "Oh, that sounds", "Hey, is anyone", "I am disappointed", "Hello,", "Hey everyone, I", "That was such", "It's okay", "The artist", "Whoa", "I understand that mistakes", "Can I help", "Who", "Hi everyone! I", "Hey, can you", "Wow, how", "Today", "Oh no, I", "Oh well, I", "Well, that", "This is the", "Yes! I finally", "Hey there little", "Hello everyone!", "Love is never", "Look at the", "This postcard", "Oh great,", "Can I", "Hmm, this is", "I understand your", "Oh, look at", "B", "I'm so", "Whoa, this", "W", "Oh, this", "Sometimes", "This piece of", "What the", "That was a", "Hey, do", "Oh no", "Whoa, what", "I feel like I", "The documentary", "Hello", "Hello little one", "I understand that my", "Eww, that", "Wow, an", "Yes! Finally,", "Although the physical location", "Whoever is watching", "That movie", "I remember wondering about", "Hey there, little", "Who's", "Hello, who", "Hello everyone! Thank", "Hello, can", "That's too", "Hey, just wanted", "Hey there, I", "Saying good", "Hey there!", "Who is there?", "Oh my good", "I am very", "Oh no, what", "Wow, thank", "I was promised", "Hi, is", "Hey, I'", "Guys, the", "Oh no, that", "Who is there", "Hello, this", "That movie really touched", "If you have something", "The documentary was", "I'm starting", "Are you kidd", "That movie really", "Hey everyone,", "Thank you for considering", "I didn'", "Yes! I", "Can you", "Oh my god", "Hey, whoever", "That melody really", "Thank you, little", "Hello, may I", "Look", "Wow, we", "It looks", "What do these", "Oh wow", "I apologize", "What are you all", "It's such", "It's clear", "Hey, I was", "Hey friend,", "I can only", "The weather outside is", "Eww, this", "I miss you", "Wow", "Aww,", "Hi, is there", "This artwork", "Okay,", "Oh well,", "This", "I'", "Say", "Hey there little gu", "Hmm,", "Whoa, who", "I am thr", "Oh man", "Okay, stay calm", "I'm happy", "Oh, this cur", "Oh man,", "I'm sorry", "Hello? Who", "What?! That", "This piece", "Hey everyone", "That's so", "Are you okay?", "What happened? Where", "Hi there", "The", "Who the hell entered", "I can", "Guys,", "What's", "What in", "It's important", "I'm", "I'm coming", "It'", "Yes! Finally", "Wait, what", "Wow, reading", "I'm surprised", "Hey, did", "Hey,", "Okay, let", "I understand that you", "Who the hell threw", "Eww, who", "Thank you for thinking", "Who is this?\"", "I am deeply", "Thank you for including", "Oh no, an", "It looks like you", "Aww", "I'm confused", "Wow, it", "That poem really", "Yes", "Hey there, is", "Hey, what'", "Thank you for remember", "To", "This is", "Thank you for making", "I can'", "That mel", "Wow, they", "I feel like", "Although the", "Who are you", "Love", "If", "What the hell are", "I am so sad", "Oh, I found", "Thank you", "It looks like", "Well, life is", "I appreciate that", "The artist's", "Whoa, that", "It's never"]