# Generates a synthetic dataset akin to the one used for internal testing
# and inside the demo.
#
# This tool relies on google's Gemma3:27b running locally with ollama for
# synthetic data generation. The dataset has been created to showcase the
# features available in BYOTA without the need to use public, albeit personal,
# data from a real social network, and this script is made available as-is
# to show how the dataset was built. The code is quite scrappy but I
# (davide@mozilla.ai) hope it can still be useful to you if you want to
# create a custom dataset.
#
# Instructions:
# - `pip install tqdm litellm` in your python env
# - run `python gen_dataset.py`
#
# The current code will generate 5 samples and dump the output as JSON.
# If you want to create a larger dataset, check out the comments and the
# code starting at line 403. What I did when generating the dataset for
# the demo was creating two JSONs, one with a smaller amount of post
# topics and a second one with more and more general topics (see comments
# starting at line 392). Example statuses are either completely made up or
# taken from my own posts.

import random
import json
from typing import List, Dict, Any, Optional, Tuple
import time
import concurrent.futures
from tqdm import tqdm
import litellm

# Parallel implementation: we are running this after starting ollama with
# OLLAMA_NUM_PARALLEL=8 ollama serve (takes ~3 mins for 32 posts)
# (TODO: test performance with different levels of parallelization)


class DatasetGenerator:
    def __init__(
        self,
        topics: List[str],
        post_types: Optional[List[str]] = None,
        tones: Optional[List[str]] = None,
        lengths: Optional[List[Dict[str, Any]]] = None,
        examples: Optional[Dict[str, List[str]]] = None,
        seed: Optional[int] = None,
    ):
        """
        Initialize the dataset generator with topics and optional parameters.

        Args:
            topics: List of topics to generate content about
            post_types: List of content types (e.g., "blog", "tweet", "review")
            tones: List of writing tones (e.g., "formal", "casual", "enthusiastic")
            lengths: List of dictionaries specifying length requirements
            examples: Dictionary mapping post types to example content
            seed: Random seed for reproducibility
        """
        self.topics = topics

        # Set defaults if not provided
        self.post_types = post_types or [
            # "blog post", "social media post", "product review",
            # "opinion piece", "news article", "tutorial",
            # "personal story", "question", "announcement"
            "social media post",
            "personal story",
            "question",
            "statement",
            "announcement",
            "introduction",
            "review",
        ]

        self.tones = tones or [
            "casual",
            "formal",
            "enthusiastic",
            "critical",
            "humorous",
            "serious",
            "technical",
            "conversational",
            "professional",
            "academic",
        ]

        self.lengths = lengths or [
            {"name": "very short", "words": (10, 25)},
            {"name": "short", "words": (25, 50)},
            {"name": "medium", "words": (50, 75)},
            {"name": "long", "words": (75, 100)},
            {"name": "very long", "words": (100, 125)},
        ]

        # Default examples if none provided
        self.examples = examples or self._generate_default_examples()

        # Set random seed if provided
        if seed is not None:
            random.seed(seed)

    def _generate_default_examples(self) -> Dict[str, List[str]]:
        """Generate default examples for each post type."""
        return {
            "announcement": [
                "I got back home from the amazing #SocialWebFOSDEM and while I am drafting a “Build Your Own Timeline Algorithm” blog post to follow up on my talk I decided to keep the momentum going with a thread about it. You'll find my code and the talk slides here. Now let's dig into #BYOTA!",
                "Weekend plan:- add Gemini support to #picogopher- build web sandbox for #ch35t- build a #brachiograph- finish reading #homeland- do some indoor bouldering - play #turingcomplete- play #projectzomboidWhat I actually did:- an as beautiful as unplanned trip to Cliveden House and Windsor- a bouldering session- read Homeland (far from finishing it though!)- read quite a lot of #manga- took two loooong baths- ate one Magnum (required for the brachiograph)",
            ],
            "introduction": [
                "Hi everyone! Six more months passed since my last #introduction, so here is an updated one:\n\nAKA: +mala, AiTTaLaM\n\nJob: Doin' trustworthy #AI @ moz://a.ai - more generally I love #teaching, no matter if to humans or machines :-)\n\nProjects: 3564020356.org is the oldest (~22yrs 😅), #PicoGopher the most recent... Look around and find the rest! 😜\n\nInterests: #bouldering #gopher #SelfHosting #opensource #reversing #fediverse #recsys #ML #solarpunk #CommunitiesOfExperience"
            ],
            "question": [
                "Hey all, are there / will there be any recordings available from the #fediversehouse event at SXSW? My FOMO is killing me 🙈"
            ],
            "social media post": [
                "Just finished my first half-marathon! 🏃‍♀️ Still can't believe I did it. Huge thanks to everyone who supported me through the training. #Running #Achievement #Fitness",
                "The sunset over Seattle tonight was absolutely breathtaking. Those moments when the sky turns into a canvas of orange and purple make you forget all the rainy days. Nature's reminder to pause and appreciate beauty.",
                "Three pints gopher://gopher.3564020356.org:70/0/phlog/2022-07-30%20-%20Three%20pints#gopher #SmallWeb #phlog",
            ],
            "personal story": [
                "Now a mirror of my gopherhole follows me in my backpack 😁 If you are in London and see this absolutely legit-looking SSID on the Tube 😅 feel free to join and browse! ❤️#gopher #selfhosting #raspberrypi #PicoGopher #WeekendProject",
                "Back on a plane for my flight back home from 3 great days at #EndSummerCamp. I am grateful for the time I spent there and for the reception of Think Smol (despite its abysmal length). I loved the people, the food, the vibes.",
                "I took many flights in my life, and I have been traveling more and more in the last few years, but I have never -and I say never- flown some taleggio cheese from UK to Italy. Not until today.I'll let you know how it goes 🙂",
            ],
            "review": [
                "I finished reading Attack Surface! It's been quite a long read for me as I had to do it during scraps of time in a difficult period, and I think I would have enjoyed it more in a different moment. Still, I am grateful it kept me company in this moment, happy I have completed the Little Brother trilogy, and I know I'll miss its characters a bit. #bookstodon #books",
                "#AmReading #bookstodon #books “Noi siamo tecnologia” (We are are technology)",
                "I am reading “How to Do Nothing: Resisting the Attention Economy”, and I am enjoying every part of it. I found about this book serendipitously, following a link from a gopherhole, and I think the way I discovered it resonates so much with its very contents!#books",
            ],
            "statement": [
                "The plural of regex is regrets :-)",
                "Meanwhile in London...",
                "... My weekend plans look more and more like OKRs: “you should not be able to always complete them, otherwise it means they were not ambitious enough” 🤦‍♂️",
            ],
        }

    def generate_prompt(self) -> Tuple[str, Dict[str, Any]]:
        """
        Generate a random prompt to feed to an LLM.

        Returns:
            A tuple containing:
            - The generated prompt string
            - A metadata dictionary tracking the selected parameters
        """
        # Randomly select parameters
        topic = random.choice(self.topics)
        post_type = random.choice(self.post_types)
        tone = random.choice(self.tones)
        length = random.choice(self.lengths)

        # Get examples for the chosen post type, or use generic examples if none available
        examples_for_type = self.examples.get(post_type, [])
        if not examples_for_type and self.examples:
            # If no examples for this specific type, use examples from any type
            all_examples = [
                ex for examples in self.examples.values() for ex in examples
            ]
            examples_for_type = random.sample(all_examples, min(2, len(all_examples)))

        # Example selection logic - select 0-2 examples
        num_examples = random.choices([0, 1, 2], weights=[0.2, 0.4, 0.4])[0]
        selected_examples = []
        if num_examples > 0 and examples_for_type:
            selected_examples = random.sample(
                examples_for_type, min(num_examples, len(examples_for_type))
            )

        # Build the prompt
        prompt_parts = [
            f"Write a {length['name']} {tone} {post_type} about {topic}.",
            f"Your content should be between {length['words'][0]} and {length['words'][1]} words.",
        ]

        # Add specific instructions for diversity
        specificity_instructions = random.choices([True, False], weights=[0.7, 0.3])[0]
        if specificity_instructions:
            specific_instructions = random.choice(
                [
                    "Include at least one personal anecdote or story.",
                    f"Mention benefits and drawbacks related to {topic}.",
                    "Compare different perspectives on this topic.",
                    "Include some specific data or statistics.",
                    "Address common misconceptions about this topic.",
                    "Start with an attention-grabbing statement.",
                    "End with a thought-provoking question.",
                    "Include a relevant analogy or metaphor.",
                    f"Reference how {topic} has evolved over time.",
                ]
            )
            prompt_parts.append(specific_instructions)

        # Add examples if selected
        if selected_examples:
            prompt_parts.append("\nHere are some example(s) for reference:")
            for i, example in enumerate(selected_examples, 1):
                prompt_parts.append(f'Example {i}:\n"{example}"')

        # Sometimes add a format instruction
        format_instruction = random.choices([True, False], weights=[0.3, 0.7])[0]
        if format_instruction:
            format_instructions = random.choice(
                [
                    "Use paragraphs to organize your thoughts.",
                    "Include a short title.",
                    "Use bullet points where appropriate.",
                    "Write in first person perspective.",
                    "Write in third person perspective.",
                ]
            )
            prompt_parts.append(format_instructions)

        # Final instruction to ensure diversity
        prompt_parts.append(
            "Avoid starting with a title or sentences like 'okay, here is...': just provide the actual content."
        )
        prompt_parts.append("Make your content unique, authentic, and engaging.")

        # Join all parts with proper spacing
        prompt = "\n\n".join(prompt_parts)

        # Create metadata dictionary
        metadata = {
            "topic": topic,
            "post_type": post_type,
            "tone": tone,
            "target_length": length,
            "num_examples_provided": num_examples,
            "prompt_creation_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        return prompt, metadata

    def _process_single_sample(
        self, idx: int, llm_function: callable
    ) -> Dict[str, Any]:
        """
        Process a single sample - helper function for parallel processing.

        Args:
            idx: The index/ID for this sample
            llm_function: Function that takes a prompt and returns generated content

        Returns:
            A dictionary with the generated sample data
        """
        prompt, metadata = self.generate_prompt()

        try:
            # Generate content using the provided LLM function
            content = llm_function(prompt)

            # Create a sample entry with all relevant information
            sample = {
                "id": idx,
                "prompt": prompt,
                "content": content,
                "metadata": metadata,
                "word_count": len(content.split()),
                "generation_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            }

            return sample

        except Exception as e:
            print(f"Error generating sample {idx}: {e}")
            return {
                "id": idx,
                "error": str(e),
                "prompt": prompt,
                "metadata": metadata,
                "generation_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            }

    def generate_dataset(
        self,
        size: int,
        llm_function: callable,
        max_workers: int = None,
        batch_size: int = None,
        with_progress: bool = True,
        retry_failed: bool = True,
        max_retries: int = 3,
        rate_limit_delay: float = None,
    ) -> List[Dict[str, Any]]:
        """
        Generate a dataset of the specified size using parallel processing.

        Args:
            size: Number of samples to generate
            llm_function: A function that takes a prompt string and returns the LLM's output
            max_workers: Maximum number of worker threads/processes (defaults to CPU count)
            batch_size: Number of samples to process in each batch (defaults to size)
            with_progress: Whether to show a progress bar
            retry_failed: Whether to retry failed generations
            max_retries: Maximum number of retry attempts for failed generations
            rate_limit_delay: Optional delay between API calls to avoid rate limiting

        Returns:
            A list of dictionaries, each containing the generated content and metadata
        """
        dataset = []
        batch_size = batch_size or size
        batches = [
            range(i, min(i + batch_size, size)) for i in range(0, size, batch_size)
        ]

        # Keep track of failed indices for retries
        failed_indices = []

        for batch_num, batch_indices in enumerate(batches):
            if with_progress:
                print(f"Processing batch {batch_num + 1}/{len(batches)}")

            indices_list = list(batch_indices)

            # Process the batch in parallel
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=max_workers
            ) as executor:
                # Create the future-to-index mapping
                future_to_idx = {
                    executor.submit(self._process_single_sample, idx, llm_function): idx
                    for idx in indices_list
                }

                # Process the results as they complete
                futures_iter = concurrent.futures.as_completed(future_to_idx)
                if with_progress:
                    futures_iter = tqdm(
                        futures_iter, total=len(indices_list), desc="Generating samples"
                    )

                for future in futures_iter:
                    idx = future_to_idx[future]
                    try:
                        result = future.result()
                        if "error" in result:
                            failed_indices.append(idx)
                        else:
                            dataset.append(result)

                        # Optional delay to avoid rate limiting
                        if rate_limit_delay:
                            time.sleep(rate_limit_delay)
                    except Exception as e:
                        print(f"Exception processing sample {idx}: {e}")
                        failed_indices.append(idx)

            # Optional delay between batches
            if batch_num < len(batches) - 1 and rate_limit_delay:
                time.sleep(rate_limit_delay * 2)  # Longer delay between batches

        # Handle retries if requested
        if retry_failed and failed_indices:
            retry_count = 0
            while failed_indices and retry_count < max_retries:
                retry_count += 1
                print(
                    f"Retrying {len(failed_indices)} failed generations (attempt {retry_count}/{max_retries})"
                )

                # Create a new list for tracking failures in this retry round
                still_failed = []

                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=max_workers
                ) as executor:
                    future_to_idx = {
                        executor.submit(
                            self._process_single_sample, idx, llm_function
                        ): idx
                        for idx in failed_indices
                    }

                    futures_iter = concurrent.futures.as_completed(future_to_idx)
                    if with_progress:
                        futures_iter = tqdm(
                            futures_iter,
                            total=len(failed_indices),
                            desc=f"Retry {retry_count}",
                        )

                    for future in futures_iter:
                        idx = future_to_idx[future]
                        try:
                            result = future.result()
                            if "error" in result:
                                still_failed.append(idx)
                            else:
                                dataset.append(result)

                            # Optional delay to avoid rate limiting
                            if rate_limit_delay:
                                time.sleep(rate_limit_delay)
                        except Exception:
                            still_failed.append(idx)

                # Update the failed indices list for the next retry round
                failed_indices = still_failed

                # Additional delay between retry batches
                if failed_indices and retry_count < max_retries:
                    time.sleep(
                        max(2.0, rate_limit_delay * 3 if rate_limit_delay else 2.0)
                    )

        # Sort the dataset by ID to maintain the original order
        dataset.sort(key=lambda x: x["id"])

        # Report final statistics
        success_rate = len(dataset) / size * 100
        print(
            f"Generation complete: {len(dataset)}/{size} samples generated successfully ({success_rate:.1f}%)"
        )
        if failed_indices:
            print(f"Failed to generate {len(failed_indices)} samples after all retries")

        return dataset


def prompt_llm(prompt: str) -> str:
    """
    Uses LiteLLM to send a prompt completion request to the specified model
    (currently ollama/gemma3:27b running on ollama)
    """

    response = litellm.completion(
        model="ollama/gemma3:27b",
        api_base="http://localhost:11434",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


def main():
    # Define your list of topics
    topics = [
        ### I used this section only for home timeline (800 posts)
        "digital rights",
        "opensource and free software",
        "books",
        "movies",
        "gopher (the protocol and the related community)",
        "bouldering",
        "permacomputing",
        "self hosting",
        "cryptography",
        "retrocomputing",
        "retrogaming",
        "fantasy consoles",
        "cooking",
        "travels",
        "school",
        "university",
        "academia",
        "parenthood",
        "working in IT",
        "working in AI",
        ### Uncomment the following for local/public timelines (1600 posts)
        # "artificial intelligence ethics", "sustainable gardening",
        # "remote work culture", "electric vehicles", "mental health awareness",
        # "space exploration", "cryptocurrency", "minimalist lifestyle",
        # "travel photography", "machine learning applications",
    ]

    # Initialize the generator
    generator = DatasetGenerator(topics=topics, seed=42)

    ### Uncomment the following block of code if you simply want to test the script
    # Generate a small sample dataset in parallel
    sample_dataset = generator.generate_dataset(
        size=5,
        llm_function=prompt_llm,
        max_workers=4,
        batch_size=5,
        with_progress=True,
        rate_limit_delay=0.2,
    )
    # Print the first sample
    print(json.dumps(sample_dataset, indent=2))

    ### Uncomment the following block of code if you want to generate the full dataset
    # # To generate your full dataset:
    # full_dataset = generator.generate_dataset(
    #     size=800,
    #     llm_function=prompt_llm,
    #     max_workers=8,  # Adjust based on your CPU cores and API limits
    #     batch_size=32,   # Process in smaller batches
    #     with_progress=True,
    #     rate_limit_delay=0.2  # Add delay to avoid API rate limits
    # )

    # # Save to a JSON file
    # with open("full_dataset.json", "w") as f:
    #     json.dump(full_dataset, f, indent=2)


if __name__ == "__main__":
    main()
