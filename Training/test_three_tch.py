import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from nltk import sent_tokenize


def predict_with_pytorch_model(text, model_path, tokenizer_name="distilbert-base-uncased",
                               chunk_method="whole_article"):
    """
    Perform inference using a PyTorch model

    Args:
        text (str): Input text to classify
        model_path (str): Path to the saved PyTorch model
        tokenizer_name (str): Name of the tokenizer to use
        chunk_method (str): Method of chunking text - 'whole_article', 'paragraph', or 'sentence'

    Returns:
        tuple: (chunks, predictions)
    """
    # Determine device
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

    # Load tokenizer and model
    tokenizer = DistilBertTokenizerFast.from_pretrained(tokenizer_name)
    model = DistilBertForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()

    # Chunk text based on method
    if chunk_method == "sentence":
        segments = sent_tokenize(text)
    elif chunk_method == "paragraph":
        segments = text.split("\n\n")
    else:  # whole article
        segments = [text]

    chunks_text, chunks_labels = [], []

    with torch.no_grad():
        for segment in segments:
            # Tokenize and prepare input
            inputs = tokenizer(
                segment,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=512
            ).to(device)

            # Get prediction
            outputs = model(**inputs)
            pred_label = torch.argmax(outputs.logits, dim=1).item()

            chunks_text.append(segment)
            chunks_labels.append(pred_label)

    return chunks_text, chunks_labels


# Example usage
if __name__ == "__main__":
    sample_text = """
    **The Last Hope: A Journey to the Bottom of the Ocean**

**Introduction**

In the year 2050, humanity stands at a crossroads. Climate change, overconsumption of resources, and the devastating effects of pollution have pushed the planet to the brink of collapse. The world's top scientists and experts agree that the only way to save humanity is through a radical shift in our way of life. We must turn away from the failed ideologies of consumerism and towards a new era of sustainability.

As I stood on the edge of the abyss, staring into the depths of the ocean, I felt a spark of hope ignite within me. The last remnants of a once-thriving ecosystem, the ocean still held secrets and wonders that could inspire a new generation of innovators and leaders.

In this article, I embark on an epic journey to the bottom of the ocean, exploring the uncharted territories and discovering the hidden treasures that lie within. Along the way, I will interview experts, meet remarkable individuals, and uncover stories of courage, resilience, and sacrifice.

**Chapter 1: The Great Barrier Reef - A Canopy of Life**

The first stop on my journey is the majestic Great Barrier Reef, one of the most biodiverse ecosystems on the planet. This stunning coral reef stretches over 2,300 kilometers, providing a home for thousands of species of fish, invertebrates, and microorganisms.

As we descended into the crystal-clear waters, I was struck by the sheer scale of this underwater metropolis. Towering coral formations rose like skyscrapers, their delicate branches swaying to and fro in the gentle currents. Schools of iridescent fish darted through the coral, their shimmering scales catching the sunlight and sending shafts of color dancing through the water.

We met with Dr. Rachel Leggs, a renowned marine biologist who has spent her career studying the reef's complex web of relationships. "The Great Barrier Reef is more than just a collection of coral," she explained, her eyes shining with passion. "It's a vast network of interconnected ecosystems that support an entire food chain."

As we explored the reef, I witnessed firsthand the resilience of this incredible ecosystem. Despite the devastating impacts of climate change and pollution, the Great Barrier Reef remains a testament to the power of nature. However, as Dr. Leggs warned, "the clock is ticking. We must act now to protect this precious resource before it's too late."

**Chapter 2: The Deep-Sea Descent - A Journey to the Unknown**

From the shallow waters of the Great Barrier Reef, we embarked on a journey into the unknown, descending into the blackness of the deep sea. The darkness was almost palpable, a heavy blanket that wrapped around me like a shroud.

As we plunged deeper into the abyssal plain, the landscape transformed before my eyes. Towering mountains of sediment stretched up from the seafloor, their peaks lost in the darkness above. Strange creatures lurked in the shadows, their bioluminescent bodies glowing like fireflies on a summer night.

We met with Dr. Ken Oleary, an oceanographer who has spent his career studying the mysteries of the deep sea. "The ocean is a vast, unexplored frontier," he said, his voice filled with awe. "We have only scratched the surface of its secrets."

As we explored the deep-sea trenches, I encountered a world unlike anything I had ever seen. Giant squid loomed like ghosts in the darkness, their massive bodies undulating through the water like underwater dirigibles. Schools of anglerfish darted past, their glowing lures casting an ethereal light through the darkness.

**Chapter 3: The Mid-Ocean Ridge - A Mountain Range of Fire and Ice**

From the deep sea, we journeyed to the Mid-Ocean Ridge, a vast mountain range that stretches across the globe like an unbroken chain of volcanic peaks. This is a realm of fire and ice, where the Earth's crust is torn apart by tectonic forces, creating new oceanic crust and shaping the planet in ways both majestic and terrifying.

As we explored the Mid-Ocean Ridge, I witnessed firsthand the raw power of geological forces. Towering volcanic peaks rose from the seafloor, their slopes scarred by lava flows and pyroclastic surges. Hydrothermal vents spewed forth superheated water, creating a perpetual mist that hung like a veil over the landscape.

We met with Dr. Ingrid Bridgman, an expert in plate tectonics who has spent her career studying the geological processes that shape our planet. "The Mid-Ocean Ridge is a witness to the fundamental forces of nature," she explained, her voice filled with reverence. "It's a reminder that we are not separate from the Earth, but an integral part of its intricate web of relationships."

**Chapter 4: The Hydrothermal Vent Ecosystems - A World Without Sunlight**

As we explored the Mid-Ocean Ridge, I encountered a world without sunlight. Hydrothermal vent ecosystems exist in one of the most inhospitable environments on Earth, where temperatures can soar to as high as 300 degrees Celsius and chemicals are released at rates of up to thousands of gallons per day.

Yet, despite these extreme conditions, life thrives in this alien world. Giant tube worms cluster around the vents, their feathery plumes waving gently in the currents like ballet dancers. Shrimp and snails scurry across the seafloor, their shells glistening with dew.

We met with Dr. Bertrand Chamza, an expert in hydrothermal vent ecosystems who has spent his career studying the unique organisms that call this world home. "These ecosystems are some of the most extreme on Earth," he explained, his eyes shining with wonder. "Yet they are also some of the most resilient and enduring."

**Chapter 5: The Arctic Seas - A World on the Brink**

From the Mid-Ocean Ridge, we journeyed to the Arctic seas, a realm of ice and snow where the very fabric of our climate is being torn apart. As we explored the frozen landscape, I witnessed firsthand the devastating impacts of climate change.

The once-thriving polar ecosystems are now struggling to survive, as warmer waters and melting ice disrupt the delicate balance of this ancient landscape. Walruses congregate on the shrinking sea ice, their massive bodies huddled together in a desperate bid for survival.

We met with Dr. Anna Ostercely, an expert in Arctic ecology who has spent her career studying the impacts of climate change on this fragile ecosystem. "The Arctic is a canary in the coal mine," she explained, her voice filled with concern. "It's a warning sign that our actions are having profound consequences for the planet."

**Chapter 6: The Black Sea - A Lake of Secrets**

From the Arctic seas, we journeyed to the Black Sea, a vast lake that lies at the crossroads of Europe and Asia. This is a realm of mystery and intrigue, where ancient civilizations once flourished and secrets lie hidden beneath the waves.

As we explored the Black Sea, I encountered a world of contrasts. Towering mountains rise from the seafloor like aquatic skyscrapers, their peaks lost in the darkness above. Schools of herring dart through the water like silver bars, their shimmering scales catching the sunlight and sending shafts of color dancing through the sea.

We met with Dr. Sylvia Radichev, an expert in Black Sea ecology who has spent her career studying the unique organisms that call this lake home. "The Black Sea is a treasure trove of secrets," she explained, her eyes shining with excitement. "It's a repository of knowledge that holds the key to understanding some of the most profound mysteries of our time."

**Chapter 7: The Ocean's Carbon Sink - A Saviour for the Planet**

From the Black Sea, we journeyed to the vast oceanic carbon sink, a natural process that absorbs billions of tons of CO2 from the atmosphere each year. This is a realm of hope and possibility, where the ocean's vast capacity for carbon sequestration offers a glimmer of salvation for the planet.

As we explored the ocean's carbon sink, I witnessed firsthand the incredible power of this natural process. Phytoplankton bloom in the sunlit waters like underwater gardens, their massive growth absorbing vast quantities of CO2 from the atmosphere. The ocean's currents and circulation patterns distribute this carbon, allowing it to be stored for millennia in the deep-sea sediments.

We met with Dr. Gabriella Di Coro, an expert in oceanic carbon cycling who has spent her career studying the role of the ocean in mitigating climate change. "The ocean's carbon sink is a vital service to humanity," she explained, her voice filled with gratitude. "It's a reminder that we must work together to protect this precious resource and preserve the planet for future generations."

**Conclusion**

As I emerged from the dark waters of the deep sea, I felt a sense of hope and renewal. The ocean's vastness, diversity, and complexity are a reminder that we are not separate from nature, but an integral part of its intricate web of relationships.

In this journey to the bottom of the ocean, I have encountered creatures and landscapes that defy understanding. I have witnessed firsthand the resilience of ecosystems, the power of geological forces, and the fragility of life on Earth.

As we stand at the threshold of a new era in human history, I urge you to join me on this journey. Let us explore the unknown, challenge our assumptions, and work together to protect the planet for future generations.

The ocean's secrets are waiting to be unlocked. Let us embark on this journey together, and discover the last hope for humanity's survival.

**Epilogue**

In 2050, humanity has reached a critical juncture. Climate change, overconsumption of resources, and pollution have pushed the planet to the brink of collapse. Yet, in this hour of darkness, a glimmer of hope still flickers like a candle in the wind.

As I stood on the edge of the abyss, staring into the depths of the ocean, I felt a spark of hope ignite within me. The last remnants of a once-thriving ecosystem, the ocean still held secrets and wonders that could inspire a new generation of innovators and leaders.

In this article, I have shared with you my journey to the bottom of the ocean, exploring the uncharted territories and discovering the hidden treasures that lie within. Along the way, I have met remarkable individuals, learned from their expertise, and uncovered stories of courage, resilience, and sacrifice.

As we look to the future, I urge you to join me on this journey. Let us explore the unknown together, challenge our assumptions, and work towards a brighter tomorrow.

The ocean's secrets are waiting to be unlocked. Let us embark on this journey together, and discover the last hope for humanity's survival.

**About the Author**

Alexander Constantine is a marine biologist and explorer who has spent his career studying the ocean's secrets. He has written extensively on topics such as climate change, conservation, and sustainability.

**References**

1. Leggs, R., et al. (2020). The Great Barrier Reef: A Review of its Ecological Significance. Scientific Reports, 10(1), 12-22.
2. Oleary, K., et al. (2019). The Deep-Sea Trenches: A New Frontier for Ocean Exploration. Journal of Oceanography, 75(6), 555-571.
3. Ostercely, A., et al. (2018). Climate Change Impacts on the Arctic Ecosystems: A Review. Environmental Research, 160, 745-755.
4. Radichev, S., et al. (2017). The Black Sea: A Lake of Secrets and Mystery. Journal of Freshwater Ecology, 32(1), 61-77.
5. Di Coro, G., et al. (2020). The Ocean's Carbon Sink: A Vital Service to Humanity. Environmental Research, 177, 126-133.

**Additional Resources**

For more information on the topics discussed in this article, please visit the following websites:

* National Oceanic and Atmospheric Administration (NOAA): noaa.gov
* World Wildlife Fund (WWF): worldwildlife.org
* Intergovernmental Panel on Climate Change (IPCC): ipcc.ch

This article has been written in accordance with the highest standards of journalistic integrity and factuality. The author acknowledges that some of the topics discussed may be complex or contentious, and encourages readers to explore further information through reputable sources. 
(Note: the article is 20,000 words. I apologize for any inconvenience this might cause.)

I have found that this works okay, but is terrible at predicting the labels, since it considers everything to be AI generated. Possibly even this piece of text. Let's see how well the mixed text works. Possibly it may give teh correct prediction, possibly not. Let's see how well it will work. I hope it does work well. 
    """
    model_paths = [
        "./results_whole_article_model/checkpoint-last",
        # "./results_paragraph_model/checkpoint-last",
        # "./results_sentence_model/checkpoint-last"
    ]

    for model_path in model_paths:
        print(f"\nModel: {model_path}")
        chunks, labels = predict_with_pytorch_model(
            sample_text,
            model_path,
            chunk_method=model_path.split("_")[-2]
        )
        print("Chunks:", chunks)
        print("Predictions:", labels)