<tip: for better readability you can press ctrl or cmd key along with plus sign key for zooming in>

# Parallel Painting - Python explanation of a CUDA parallel painting algorithm

Welcome to my first article!<br>
It is about a parallel painting algorithm I developed some years ago and implemented in CUDA, executed on an NVIDIA GTX card. Although I hadn't found anything similar when looking for a parallel painting solution, it is more like I reinvented the wheel. As you will see, it is more like a traditional 'log2N' algorithm than a big invention.

In this article, I will talk about this algorithm, show how it works with example animations, and show how we can execute it on some paper with colored pencils. Don't worry if you are not an expert in math or programming. I will try to explain everything thoroughly and simply.

Instead of the original CUDA and C++ source code, which runs only on NVIDIA GPUs, I will also present some code in Python. Installing and using Python is fairly easy on any computer or handheld device. I was working on this article on my old-style MacBook Air, including all the Python work.
I'm also planning a follow-up article, which would be about the CUDA algorithm itself and would answer some questions I leave open here (data races, race conditions, these topics are more complicated).

Before the parallel algorithm, I will show a traditional painting algorithm working on a single processor core.

### Color islands
<img align="Right" src="./pictures/color_islands.png" alt="colorful islands in a light blue sea" width = "40%">


Hopefully, the image on the right looks like some fancy-colored islands in a calm sea. At least, that was the goal when I tried to use my finger-painting skills on a small smartphone touchscreen.
I could draw black island outlines on a white screen with a simple application. The app had a paint function, which showed a tilted paint bucket filled with paint. I could move the bucket around with my finger and "pour out its content" with a button on the bottom of the screen. The application flooded all the white parts with light blue when I poured the carefully selected color for the sea outside all island contours. Not yet knowing the colors of the islands, I moved the bucket inside all of the islands and poured black paint into all of them. This way, I could remove the outlines as the islands turned all black.

&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;

<img align="Left" src="./pictures/black_islands.png" width = "40%">
<br>
I can not say I had a clear plan about the colors of the islands, but I started to move around the bucket, set the color of its paint, and tapped the 'pour out' button when the bucket was over a black area. The application relentlessly turned the black islands into colored ones.
The painting process looked instantaneous. The paint flooded an island in a blink of an eye. It is quite convenient from the user's point of view. However, it is not quite realistic. At least all paint or liquid I've ever seen was flowing way, much slower than the speed of light (hopefully, I didn't miss something, and this is always the case).
I guess you wouldn't be so surprised if I told you that the "paint" in the application also flows in some direction or directions, flooding everything until it hits the sea, which acts as a wall. It either stops the paint in that direction or, more probably, redirects it to other still black areas in the neighborhood.
Sounds pretty complicated! Thankfully or sadly, even small devices like smartphones are fast enough to do this complex task immediately.
If you are curious about how this painting process works under the hood, you are just like me! If not, further reading of this article can bring up your interest.
I don't know how the particular paint filling in the app works, not least because it paints instantly, but I have hazy memories from my childhood when things were slower, especially computers. You could easily follow the painting process, sometimes for tens of seconds. If I remember correctly, they looked more like brush strokes. Pixels were drawn up and down between the outlines, stepping one pixel to the sides after each vertical stroke.
I'm sure it painted pixels one by one, strictly one at a time. It is understandable, as processors could do only one thing at a time. They had single cores only. Nevertheless, designing an algorithm for one core is usually simpler. How could 8 or 16 cores help us out here? Can a brush stroke draw 8 pixels at once? What if one of the pixels hits the outline? How would the other pixels know they would fall outside?
Similarly, could we have 16 brush strokes at once? These are some of the questions that came to my mind. Well, I'm sure there are clever answers to them, but I'm afraid they are rather complicated. So I recommend we stick to a one-step-at-a-time method when thinking about a simple solution.
Now I can hear you asking why I'm talking here about a simple one-thread (one core) solution when you already read the title and preface of this article. Spoiler alert: yes, I will introduce my parallel algorithm later on in this article, which runs on thousands of cores of a GPU at once (!), manipulating thousands of pixels at a time! That will be another story. It will look more like a flood fill. First, let me introduce a classic, one pixel at a time, brush stroke-like algorithm to see the contrast between the two.

### A single-core painting algorithm

Thinking back to old brushstroke algorithms, I don't remember how they filled entire shapes with pixels. Just imagine we specify the starting point inside a rectangle shape. Pixels appear upwards with a given color, then downwards one pixel to the left, then up to the left again until we reach the left side of the shape. But what about the other side, the right one from the starting point? If I remember correctly, after these algorithms painted the left side, pixels started to appear downwards from the starting point, then upwards one pixel to the right, all the way to the right. It was a kind of 'restart' from the starting point. Now you can imagine what it is like to have a more complex shape. Such as a rotated or lying **H** shape, one thin vertical rectangle with two thin horizontal rectangles on its bottom and top. That must have 3 'restarts' from 3 different pixels. It doesn't sound like a simple algorithm. It must be quite complex!<br>
I was already thinking about this process for this article when I asked my lazy programmer self about his (my) opinion. It is complex, but let's outsource the complex 'restart' method to the machine!
Let's start with easier things. Moreover, let's start at the very beginning! We need a pixel coordinate to start somewhere inside a black island on an image and a color for drawing pixels.
We could write this Python function header:
```python
def paint(image, x, y, color):
```
In the first step, can we simply paint the given pixel with the specified color? Surely, but we need to be cautious. If the pixel at the given coordinates is not black because it is a blue sea pixel or an already painted island pixel, we should not paint it. Also, we can not start the painting process. In this case, we will simply return from the paint function. Otherwise, we can paint the pixel to the given color.<br>
If you feel something is missing, an error message for the failure case, or the sanity check of the x and y coordinates, your intuition is right! However, I would skip the sanity check for simplicity, and I should skip the error message for reasons you are about to see.
So, if the given coordinates point to a pixel inside a black island, we can paint the first pixel. So far, so good! But what about the others? Should we go up one pixel, down, or what? Now let's see the simple approach of my lazy programmer self.
Where are the remaining black pixels? Around the pixel, we have already painted! Great situational awareness, but what can we do with this information? Let's say we never go too far, inspecting only the pixels around the already-painted pixel. Ideally, they are all black, or at least most of them. Let's try to paint all the neighbors around the painted pixel individually. We take one neighbor by its coordinates; if it is not black (not already painted or sea pixel), we don't do anything with it. If it is black, we will paint it in the given color.
Sounds familiar? Yes, we have just done the same with the first pixel! And here comes the trick: after we paint a pixel, we always take all its neighbors again and do the same with those pixels individually. If possible, paint it to the given color and take its neighbors one by one again.
I know it sounds ridiculous, mostly because one of the neighbors of a pixel neighbor is the pixel itself, which is already painted (I hope I haven't lost you at this point, and you are still with me). It is not a problem since we must always check every pixel before painting.
As the task is the same for the first pixel and all the pixels around it, we can simply call the paint function itself inside the paint function for all neighboring pixels. Calling a function inside the same function may sound like a recipe for a trouble, coming in the form of a nice infinite loop, and it certainly can be if a function always calls itself. However, we already have a case when we don't need to check the neighbors. It is when the pixel we are working on is an already painted or a sea pixel. In that case, we return from the function without calling it again.

Sounds good! Let's see the whole function in Python!

```python
def paint(image, x, y, color):
    if image[y,x] == black_color:
        image[y,x] = color
        paint(image, x    , y - 1, color) # neighbor above
        paint(image, x    , y + 1, color) # below
        paint(image, x - 1, y    , color) # to the left
        paint(image, x + 1, y    , color) # to the right
        paint(image, x - 1, y - 1, color) # top left neighbor
        paint(image, x + 1, y - 1, color) # top right
        paint(image, x - 1, y + 1, color) # bottom left
        paint(image, x + 1, y + 1, color) # bottom right
    return
```
> **_Note:_**
> - `image` is a 2D numpy array of color code numbers, the image of islands
> - `color` is a color code number between 0 and 11, 0 means black, 1 is the sea color, and numbers between 2 and 11 represent island colors
> - `black_color` is 0
> - as usual, x is the horizontal axis, and y is the vertical
> - indexing a 2D numpy array is a bit tricky: y comes first, x is the second
> - numpy arrays are mutable, which means any modification of the array inside the function is also seen from outside (passed by reference in C++ terms)



You can see 8 self-function calls inside the function, as all pixels have 8 neighbors.
Smart people have already named these kinds of self-calling functions recursive because they recursively call themselves most of the time. Don't worry if you are unfamiliar with this concept. Don't immediately trust anyone who claims to know how and when the calls and returns occur in a small but rather complicated recursive function like this one. As in our case, this is often not known in advance since it depends on what data the function is working on. In our case, this data is the island's shape and the starting point's location. The machine automatically handles the path of function calls and returns. With recursion, we only need to consider the maximum depth: the number of times the function calls itself before a return happens. In our case, it can be a few thousand, which is more than the thousand recursion depth allowed by default in Python.<br>We can easily change this upper limit:

```python
import sys
sys.setrecursionlimit(4000)
```

Let's see how this recursive paint function works. I mean, literally see it!

<img align="Left" src="./pictures/island_paint_1st_of_2.gif" alt="black island is being painted" width = "40%">

The first thing that stands out is that this process resembles what I was talking about, recalling my memories. Not always, but most of the time, the pixels go up and down, mimicking up and down brush strokes, even if there is no explicit code to do so in the function. If this doesn't surprise you, you are already an expert in recursive functions! As I've already stated, the recursion path heavily depends on the data, the island's shape, and the starting point's location. However, we can easily influence the process by the order of the function calls, that is, the order of neighbors to inspect. I've intentionally placed the investigation of the upper and lower neighbors first. Remember, we have only one core here. The function can cope only with one pixel at a time. Instead of taking the neighbors of a pixel one by one, the process takes the first neighbor, the upper one, then the upper neighbor's first neighbor, which is its upper neighbor (assuming those neighbors are still black). That results in the brush stroke up. If it can not proceed up, the second function call kicks in, resulting in a brush stroke down. You can also see the "restarts," they come automatically. These restarts can occur when many functions return from each other because they visited all neighbors in a direction. Visually, this is when a part of the island is already fully painted, and the brushstroke can not continue in that area, in that direction.

&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;

Let's see what happens if the order of the calls is different. In this scenario, we would start with the top-left neighbor and then proceed clockwise.
> _**fun fact:** This was my original attempt at this recursive paint function. Later I rearranged the calls as I progressed with this article and started writing about old-school algorithms._


<img align="Right" src="./pictures/island_paint_2nd_of_2.gif" alt="black island is being painted" width = "40%">

Not surprisingly, the first brush stroke goes diagonally up to the left. Then it turns to other, seemingly unpredictable directions.

The second thing that is quite obvious is that this painting process is slow. Here it takes minutes, although the pixels appear quickly one after another. I set the pixel rate to 30 pixels/second for these demonstrations. 3 pixels in one-tenth of a second! However, the overall process looks slow. It is because we have many pixels on an island. The whole image of islands has 280x372 pixels. An island would roughly fit in a 100x100 pixel square. 100 doesn't seem like a lot, but its square is another order of magnitude; it is 10 000 pixels. The first island of the demonstration consists of 5758 pixels, and the other one is 6696 pixels. Their width and height have a similar order of magnitude. Let's denote it with $n$. Their square is another order of magnitude. We can denote it with $n^2$. If we see an island twice as tall and wide as another, we perceive it as twice as large, even though it has 4 times as many pixels. An island that looks 20 times bigger has 400 times more pixels!

&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;

Even with a fast machine that can process 100 000 pixels a second, it would take roughly 1 second to paint all the islands on the black islands picture. Processing $n^2$ pixels is simply too much, considering we can only deal with one pixel at a time. We can say that an algorithm that executes a task $n*n$ times is too time-complex. There is even a special notation that tells how time-complex an algorithm is. It is the big O notation, denoted either with a simple capital O or with a fancier open O at the top:

$$O(n^2)$$
$$\\mathcal{O}(n^2)$$

$O(n^2)$ algorithms are quite slow. They need optimization if possible. $O(n*log(n))$ complexity is usually a reachable goal for them and is better, $O(n)$ is even better, and $O(log(n))$ algorithms are pretty fast. There are also $O(1)$, constant time algorithms or operations. They are the fastest ones. On the other hand, we have very slow algorithms such as $O(n^3)$, $O(2^n)$, or even $O(n!)$, to name a few (I think permutation is a good example for the latter).<br>
In our case, the only possibility to speed up the painting algorithm is to process more than one pixel in one step. It is what we are going to do in the parallel painting algorithm. However, that will be a different algorithm. Parallelizing our recursive function would be inefficient for many reasons.

> _**fun fact:** Although the time complexity of my parallel algorithm will be better, the algorithm itself will be more complicated, consisting of several steps and more than one function. It is quite common. Parallelization always involves some overhead._

That may sound like a recipe for trouble, and it certainly can be if we are not careful. <- quote from Barron Stone
$$ O(n^2) $$
$$ O(log_2(n)) $$
$$ \\mathcal{O}(log_2n) $$
&#119978; $$ (log_2n) $$

### The parallel painting algorithm

Originally written in CUDA and C++, my parallel algorithm ran on GPUs with a few thousand cores. As I mentioned, we will still use Python in this article, so although the algorithm will be the same, it won't run in parallel.

The original CUDA algorithm is a short chain of certain steps. These steps are executed serially. In each step, functions are executed on the GPU (CUDA kernels in CUDA terms). I created two functions. They are executed alternately during the iteration. The only exception will be the first step of the iteration when only one of the functions is called. Each function has as many instances on the GPU during processing as many pixels we have on the image. It also means that all islands will be painted at once! In theory, all pixels are processed in parallel with these functions, more precisely, with their instances on each core. I will tell you more about the practice later on.

Now you may wonder how to assign starting points and different colors to the islands if we paint them all at once. With this algorithm, no starting points need to be assigned, and we don't need to specify the color before the painting process. The algorithm fills the islands with arbitrary numbers. These numbers will be the same on one island but completely different from those on the other.

Enough talk, let's see what it looks like!

<img src="pictures/island_paint_parallel.gif" alt="salt and pepper colorful islands turn to plain color ones in a few steps, as colorful salt and pepper grow bigger and bigger" width = "40%">

As you see, the result island colors are not the same as the ones on the original image. Moreover, some islands have the same color at the end. It is because the numbers are arbitrary, which fill each island. Their value can be between 1 and the number of all the picture pixels (280x372 = 104160). I have just a few colors to represent these numbers, so I used a simple modulo operation. 10 colors are repeated in this 104160-long interval. It is the reason for the same color islands, but I can assure you their pixel values are different. After these iteration steps, the result island values can be easily mapped to our color codes to "paint" them to their original colors. I put the word paint in quotation marks, as it is more like a simple value substitution operation now. We need to do it for a bunch of pixels, but we can do it at once in one step. On GPU, all pixels can be substituted at once. In Python, it will be one simple command for each island. If colors are irrelevant, this last step can be omitted, or all the island values can be automatically gathered and mapped to different color codes. If colors matter, as in our example, we can use the method I showed in the single-core case, manually giving each island an x, y, and color value.


Now let's see how this iteration with parallel steps works in detail! Let me use a real notebook and real pencils to explain the steps of the algorithm. For the sake of simplicity, the toy example here will be my original first example with only 6x6 pixels and one island first.

<img src="./pictures/toyexample_small.jpeg" alt="small pencil island with small pencil sea" width = "20%">

>_**fun fact**: I was drawing and writing many numbers in a similar notebook when I came up with this algorithm. The one-island example here will be the same as the one I came up with years ago. It happened after many tries, and even with this example, it took me a while to get the algorithm right._

First, I draw a grid that represents the 6x6 image. I number all the pixels in the upper left corner with a blue pen in ascending order. I apologize if I refer to these blue numbers as slot numbers (I drew the pixels so big they look like slots).

Now comes a trick: the first preparatory step before the paint iteration. Let's write these numbers in big black in the pixels representing the islands (black pixels considering the examples of the single-core version). The sea pixels remain blank, but since I bought a fancy teal pencil, I would shade them with sea color. Note that instead of black pixels, each island pixel has a separate number in an ascending order.

During the iteration, we will work on certain areas of the image. We start with small areas and increase their sizes as the iteration progresses. I draw these areas with my red pencil. As you see in the picture, these areas also form a grid. An area is 2x2 pixels large in the beginning.

For this demonstration, I will show the two functions of the iteration steps as two substeps.
In the original CUDA algorithm, all the pixels are processed simultaneously. As we can not do the same in the Python version, and I'm only good with one hand and pencil, I will do the algorithm tasks for one pixel at a time. The order doesn't matter. However, in the Python implementation, I use the order of the slot numbers. I will use the same order in the notebook for these demonstration videos and cover some interesting things about parallel or different-order scenarios.

Let's start the iteration with the first substep to do. We take all the island pixels and inspect their island pixel neighbors within the red area. It is important to take the only neighbors into consideration which are located in the same area. If the smallest value around the island pixel is less than the pixel's value, we take note of it.

I will write down all the pairs of pixel values and smaller neighbor values with a $<$ sign, as you see in the demonstration video. I will refer to the values on the two sides of the $<$ sign as left and right.

After having all the individual pairs, we can finish the first substep. I take all the value pairs written down one by one. The larger right numbers show the slots we need to modify. I take a pair, find the slot number according to the right value, and replace the pixel value in the slot with the smaller left value of the pair. I know it sounds a bit cumbersome because the slot numbers and the pixel values in the slots are the same. However, it will not be the case in later steps. Thus, it is important to see the right side values of the pairs as slot numbers (in fact, the left values also refer to slot numbers, but their pixel values will always be the same as their slot values).

It is worth mentioning that even in our Python implementation, these seemingly two subtasks (identifying pairs, *and then* modifying values with them) will be done in one step for each pixel. In the Python implementation, we take all pixels individually, check their neighbors, and immediately modify the small value slot if there is a smaller one.
In this iteration step, this sounds good enough to do so, but in later steps, this can and will raise questions.

> __thoughts for the parallel version:__ !!TODO!! NOT here, in later steps!

And we are done with the first iteration step! As I mentioned earlier, we silently skipped here a second substep which could have also been performed for the first iteration step. However, it would have had no effect. Let's go on with the second iteration step, where I will also reveal the second substep and why it is unnecessary for the first iteration step.

For the next iteration step, we double the sides of the areas. Now they are 4x4 pixels in size; see my red lines. Note that almost all areas except the first one stick out since the image is only 6x6 pixels, but that's not a problem. I even left them open on my drawing in the notebook, as it doesn't matter. The rules are the same for the first task. We are looking for the smallest value neighbor of each pixel. If the smallest neighbor is smaller than the pixel value, we note it. Only values in the same area as the pixel can be neighbors.
We evaluate our notes the same way as in the first iteration step.

!!! questions arisen here, root values here

Although the task is the same as in the first iteration step, many interesting things have revealed their true face. As mentioned earlier, right-hand values in our notes must be considered slot numbers. Now it makes sense. See the values 13, for example, in slots 13, 14, 19, and 20. We don't change all of them to 1 just because we have the note 1 < 13. We change only one of them, which is in slot 13. Let me call the value in slot 13 the 'root' for all 13 values. This root changes from 13 to 1 and will be the new root for all values of 13. If you are wondering why don't we change all 13 values to 1, only the root, I can assure you that they will all be changed. However, it is not done by the first substep. It will be handled by the second one, which is about to be introduced. Remember, in implementing the algorithm, these two subtasks (take notes, then evaluate them) are done in one step for each pixel. It would be slow to take and evaluate notes and cost extra memory. Changing all 13 values and checking for neighbors in one parallel step would present additional challenges.

TODO: ideologies later please
The best summary for the first task is to change the root of !TODO think it over: root of what? similar values? from when they exist?

And with this, I've already revealed the sole goal of the mysterious second step. This substep takes all pixel values and changes them according to their roots. Its way of working is quite simple. We take each pixel's value and look up its root, the value in the slot with the same number as the pixel value. I can imagine your eyes widening after reading the last sentence. Hopefully, it sounds more fun than complicated. Don't worry if the latter is the case. I'm trying to demonstrate this root lookup thing with my pencil. Hopefully, this makes it clearer (at least a bit). If it is clear, I'm ready to increase this substep's funniness (or the complexity). After finding the pixel's root value, we will not immediately replace the pixel value with the root value. First, we ensure that the root value and the slot number of the slot where the root value resides are the same. If not, we look up the root value of the root value we already found. Anybody still with me and finished her or his loud laughter can see this more easily with values borrowed from my example. Let's say we process the pixel in slot 19. It has the value '13'. Now we focus on slot 13, which has the value 1. Slot number 13 and its value 1 are not the same, so we should focus on slot 1 (as the root value in slot 13 was 1). Slot number 1 has the (root) value 1. The two numbers are the same. After all this, we can safely modify the pixel value of slot 19 from 13 to 1. See the appendix for more explanation if you find this last step with looking up slot 1 is unnecessary.


> **Note:** If you can't find the proper definition of 'root' in the article, it is my fault: I did not give one. Let's say all pixels have a root slot with the same slot number as the pixel value. The root slot, as an ordinary pixel, also has a value. I named this value as the root value. Note how the root values are the same as pixel values initially when we put the initial increasing numbers into our black islands.

