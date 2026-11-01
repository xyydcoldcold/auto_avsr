"""Build the personal training prompt list (train_prompts.csv) for the LoRA.

Usage (from the repo root):
    python silent_speech/make_train_prompts.py

Sources: every phrase in phrases.txt that is not a test prompt (chat messages, the target use),
plus general everyday sentences so the adapter doesn't only learn chat phrases. No sentence from
prompts.csv (the fixed test set) is ever included. The order is shuffled once with a fixed seed;
record in file order, so "the first k minutes of training data" is always the same subset.
"""
import csv
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent

GENERAL = """
the kettle is boiling
my shoes are by the door
she painted the fence blue
we planted tomatoes in may
the dog chased the ball
he fixed the broken chair
the cat is asleep on the sofa
i need a new phone charger
the bread is still warm
my sister lives in boston
the bus stop moved last week
he plays guitar on weekends
the window was left open
our neighbors have a new puppy
the soup needs more salt
i lost my umbrella yesterday
the printer is out of ink
she runs every morning
the elevator is broken again
we ordered too much food
the baby is finally asleep
my laptop battery died
the park is full of people
he forgot his jacket
the sky is very clear tonight
the shop opens at nine
i finished the whole book
the floor is still wet
she loves spicy food
we drove to the coast
the paint is not dry yet
my keys were in my pocket
the meeting room is booked
he speaks three languages
the fridge is almost empty
i walked home in the rain
the tea is too hot
they won the game last night
the flowers need water
my glasses are on the table
the hotel has a pool
she wrote a long letter
the street is very quiet
we missed the last bus
the music is too loud
my bike has a flat tire
he works from home on fridays
the cake is in the oven
i can hear the birds
the movie starts at eight
she called her mother
the box is very heavy
we need more chairs
the light in the hall is out
my phone is on silent
he likes his coffee black
the plane landed early
i cleaned the kitchen
the door is stuck
she found a good parking spot
the river is very high
we watched a documentary
my hands are cold
he lost his wallet on the train
the shirt is too small
i bought fresh fruit
the clock is five minutes fast
she started a new job
the road is closed for repairs
we had pasta for dinner
my back hurts today
he moved to a new apartment
the internet is slow tonight
i forgot to lock the car
the soup is ready
she is learning to swim
the store was very crowded
we booked a table for four
my brother is visiting next month
he wears a blue hat
the grass needs cutting
i made a cup of tea
the picture is on the wall
she missed her flight
the battery is fully charged
we walked along the beach
my camera is broken
he cooked a big breakfast
the glass is half full
i need to buy stamps
the dishwasher is running
she sings in a choir
the kids are watching cartoons
we painted the bedroom
my alarm did not go off
he left a note on the door
the shower is very cold
i will wash the car tomorrow
the bag is under the desk
she reads before bed
the garden looks lovely
we found a quiet cafe
my headphones are tangled
he takes the bus to work
the bill came to forty dollars
i heard a strange noise
the towels are in the closet
she bought a red dress
the heater is not working
we are painting the fence
my cousin is getting married
he answered the phone
the room is too warm
i need a haircut
the oven takes ten minutes to heat
she drives a small car
the photos look great
we stayed up too late
my plant is growing fast
he dropped his phone
the table is set for dinner
i like this song a lot
the trash goes out on monday
she keeps a diary
the hill is very steep
we took the stairs
my computer needs an update
he brought his own lunch
the candle smells nice
i ate too much cake
the noise woke me up
she has a big family
the car is in the garage
we rented a small boat
my tooth hurts a little
he is good at chess
the pizza is cold
i put the milk back
the door bell is ringing
she likes green tea
the room has a nice view
we bought a new rug
my socks do not match
he sent me a postcard
the pool is closed today
i fell asleep on the couch
the cookies are in the jar
she lost her earring
the line was really long
we fed the ducks
my friend is a nurse
he likes old movies
the air feels fresh
i will clean my room
the sink is full of dishes
she ran for the bus
the fan is too noisy
we sat by the fire
my ticket is on my phone
he grew a beard
the shoes were on sale
i need some fresh air
""".strip().splitlines()


def main():
    with open(HERE / "prompts.csv", newline="", encoding="utf-8") as f:
        test = {" ".join(r["text"].lower().split()) for r in csv.DictReader(f)}
    chat = [p for p in (HERE / "phrases.txt").read_text(encoding="utf-8").splitlines() if p.strip()]

    rows, seen = [], set(test)
    for scene, texts in (("daily_messages", chat), ("general", GENERAL)):
        for text in texts:
            text = " ".join(text.lower().split())
            if text and text not in seen:
                seen.add(text)
                rows.append({"split": "train", "scene": scene, "text": text})
    random.Random(0).shuffle(rows)

    with open(HERE / "train_prompts.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "split", "scene", "text"], lineterminator="\n")
        w.writeheader()
        for i, r in enumerate(rows):
            w.writerow({"id": f"t{i:03d}", **r})
    n_chat = sum(r["scene"] == "daily_messages" for r in rows)
    print(f"{len(rows)} training prompts ({n_chat} chat, {len(rows) - n_chat} general), none from prompts.csv")


if __name__ == "__main__":
    main()
