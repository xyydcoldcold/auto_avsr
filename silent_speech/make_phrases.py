"""Build the closed phrase set (daily short messages) used by the closed-set check.

Usage (from the repo root):
    python silent_speech/make_phrases.py

Writes silent_speech/phrases.txt: one phrase per line, lowercase. It always contains the
50 in-scene test prompts and never the open / out-of-scene prompts, which serve as
"not in the list" examples. Hand-written phrases plus a few templates (times, places,
meals) so the list has realistic look-alikes such as "see you at five" / "see you at nine".
"""
import csv
from pathlib import Path

HERE = Path(__file__).resolve().parent

HANDWRITTEN = """
ok
okay
yes
no
sure
yep
nope
maybe
thanks
thank you
thank you so much
thanks a lot
thanks for everything
no problem
you're welcome
sorry
i'm sorry
so sorry
my bad
all good
sounds great
sounds good
that sounds fun
perfect
great
awesome
nice
cool
got it
noted
will do
on it
done
me too
same here
agreed
of course
absolutely
not really
not yet
not sure
i don't know
i have no idea
let me think about it
let me check
let me ask
i'll ask her
i'll ask him
i'll let you know
keep me posted
let me know
let me know if you need anything
just let me know
what's up
how are you
how are you doing
how's it going
how was work
how was school
how was the trip
how did it go
how are things
what are you doing
what are you up to
where are you
where are you going
where should we meet
when are you coming
when do you get off work
when will you be home
when is it
what time is it
what time should we meet
what time do you finish
who is coming
who else is coming
is everything okay
are you okay
are you home
are you awake
are you busy
are you coming
are you sure
are you ready
are you still coming
can you talk
can you hear me
can you help me
can you call me
can you text me
can you send it again
can you wait for me
can you come over
can you grab some milk
can you pick up dinner
can you pick up the kids
can you open the door
can i call you
can i call you later
can i come over
can i borrow your car
do you need anything
do you want anything
do you want to come
do you want to get coffee
do you have time
do you have a minute
did you eat
did you see that
did you call
did you get home safe
did you get it
i'm here
i'm outside
i'm downstairs
i'm in the car
i'm on the train
i'm on the bus
i'm at work
i'm at home
i'm at the gym
i'm at the store
i'm still at work
i'm still in the meeting
i'm stuck in traffic
i'm running late
i'm a bit late
i'm so sorry i'm late
i'm on my way home
i'm heading out
i'm heading home
i'm leaving work now
i'm almost home
i'm almost done
i'm done
i'm ready
i'm tired
i'm hungry
i'm sick
i'm feeling better
i'm free tonight
i'm busy right now
i'm busy today
i'm driving
i'm going to bed
i'm going to sleep
i'm awake
i'm just leaving
i'm getting ready
i'm in the shower
i'm at the doctor
i just woke up
i just landed
i just got here
i just got your message
i just saw this
i missed your call
i'll call you back
i'll call you later
i'll call you tonight
i'll be right there
i'll be home soon
i'll be home late
i'll be late
i'll be ready soon
i'll be there in a bit
i'll meet you there
i'll pick you up
i'll text you when i'm there
i'll send it now
i'll check
i'll try
i'll see you there
i'll wait for you
i'll be down in a minute
i can't talk right now
i can't make it
i can't come tonight
i can't find it
i can make it
i can do that
i can pick you up
i can't wait
i miss you
i love you
love you too
miss you too
i got it
i got your message
i got home
i got the tickets
i need help
i need a ride
i need to go
i need more time
i have to go
i have a question
i have a meeting
i have class
i forgot
i forgot my phone
i forgot my wallet
i lost my keys
i left my keys at home
i think so
i don't think so
i hope so
i agree
i understand
i see
i know
i didn't know
i'm not sure yet
we're here
we're on our way
we're almost there
we're running late
we're at the restaurant
we're leaving soon
we're out of coffee
we're out of bread
we need milk
we need to talk
we should hang out
we should get dinner
we can talk later
we'll see
see you
see you soon
see you later
see you tonight
see you next week
see you there
see you in a bit
see you on monday
see you this weekend
talk soon
talk later
talk tomorrow
catch you later
bye
goodbye
good luck
good job
well done
have fun
have a good day
have a good night
have a safe trip
have a nice weekend
enjoy
enjoy your meal
get well soon
feel better
sleep well
sweet dreams
drive safe
be safe
take your time
no rush
don't worry
don't worry about it
never mind
it's fine
it's okay
it's not a big deal
that's fine
that's great
that's okay
that's awesome
that's too bad
that's so funny
that's right
that's not right
that makes sense
that's perfect
works for me
fine by me
either is fine
up to you
whatever you want
your call
call me
call me back
call me later
call me when you're free
text me
text me when you get home
text me when you're done
send me a picture
send me the link
send me your location
send me the details
send it to me
share your location
check your email
check your messages
answer your phone
pick up the phone
come home
come over
come with me
come downstairs
come pick me up
wait for me
wait a second
hold on
one second
just a minute
almost ready
be right back
back in five
meet me outside
meet me at the entrance
meet me at the station
let's go
let's meet
let's meet at noon
let's talk later
let's do it
let's get coffee
let's get pizza
let's order food
let's go out tonight
let's watch a movie
let's stay in
how about tomorrow
how about friday
how about next week
what about you
what do you think
what do you need
what should i bring
what should we eat
what happened
what's wrong
what's the address
what's the plan
what's for dinner
where's the party
happy new year
merry christmas
happy anniversary
happy mothers day
happy fathers day
good afternoon
good evening
morning
night
hello
hi
hey
hey there
long time no see
nice to meet you
nice to hear from you
it was great to see you
it's been a while
proud of you
congrats
you're the best
you're right
you too
same to you
i'll be online later
my phone is dying
my battery is low
my train is late
my flight is delayed
the bus is late
the train is delayed
there's a lot of traffic
traffic is terrible
parking is a nightmare
it's raining
it's cold outside
bring an umbrella
dinner is ready
food is here
the door is open
the door is locked
i'll leave the door open
the meeting is over
the meeting ran late
the meeting got moved
can we push it back
can we do it later
can we talk tonight
can we meet tomorrow
can we move it to friday
let's reschedule
something came up
i'm double booked
count me in
i'm in
i'm out
not tonight
another time
rain check
maybe next time
""".strip().splitlines()

NUMBERS = ["five", "ten", "fifteen", "twenty", "thirty"]
CLOCK = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "noon"]
WHEN = ["today", "tonight", "tomorrow", "this weekend", "on friday", "on saturday", "on sunday", "next week"]
PLACES = ["coffee shop", "office", "station", "airport", "mall", "park", "library", "gym", "hospital", "restaurant",
          "grocery store", "bus stop"]
MEALS = ["lunch", "dinner", "breakfast", "coffee", "a drink", "brunch"]
PEOPLE = ["mom", "dad", "the kids", "your brother", "your sister"]

TEMPLATES = (
    [f"running {n} minutes late" for n in NUMBERS]
    + [f"be there in {n} minutes" for n in NUMBERS]
    + [f"i will call you in {n} minutes" for n in NUMBERS]
    + [f"see you at {c}" for c in CLOCK]
    + [f"meet you at {c}" for c in CLOCK]
    + [f"see you {w}" for w in WHEN]
    + [f"are you free {w}" for w in WHEN]
    + [f"call you {w}" for w in WHEN]
    + [f"i am at the {p}" for p in PLACES]
    + [f"meet me at the {p}" for p in PLACES]
    + [f"on my way to the {p}" for p in PLACES]
    + [f"let's grab {m} {w}" for m in MEALS for w in ["today", "tomorrow", "this weekend"]]
    + [f"want to get {m}" for m in MEALS]
    + [f"can you call {p}" for p in PEOPLE]
    + [f"tell {p} i said hi" for p in PEOPLE]
)


def load_prompts():
    with open(HERE / "prompts.csv", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    prompts = load_prompts()
    test_in = [p["text"] for p in prompts if p["split"] == "in_scene"]
    test_out = {p["text"] for p in prompts if p["split"] != "in_scene"}

    phrases, seen = [], set()
    for text in test_in + HANDWRITTEN + TEMPLATES:
        text = " ".join(text.lower().split())
        if text and text not in seen and text not in test_out:
            seen.add(text)
            phrases.append(text)

    (HERE / "phrases.txt").write_text("\n".join(phrases) + "\n", encoding="utf-8")
    print(f"{len(phrases)} phrases ({len(test_in)} in-scene test prompts included)")


if __name__ == "__main__":
    main()
