# 🎨 **Customization Guide**

This Blueprint is designed to be flexible and easily adaptable to your specific needs. This guide will walk you through some key areas you can customize to make the Blueprint your own.

---

## 🧠 **Changing Embedding Model / Engine**

BYOTA currently supports [both llamafile and ollama](https://mozilla-ai.github.io/byota/getting-started/#step-1-running-a-local-embedding-server)
as embedding engines. This means you can choose to run either as long as you provide the correct server
URL. Llamafile follows the OpenAI API, so in theory other engines which are compatible with it should
work with BYOTA.

The embedding model used by default is [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2).
If you want to try a different one, you can follow the instructions provided for the engine of your choice
(pointers to documentation for llamafile and ollama are already available [here](https://mozilla-ai.github.io/byota/getting-started/#step-1-running-a-local-embedding-server)).


## 📝 **Using a different data source**

Except for the code that downloads timeline and personal posts, BYOTA is platform-agnostic so you can just plug in a different client and get recommendations for your own favorite platform. As long as you have a pandas dataframe to
pass as an input you can work with synthetic data (as we did in the demo), feeds from other social networks, article
summaries from RSS feeds... Your imagination is the limit!


## 💡 Other Customization Ideas

Do you want to run different algorithms or create a whole new application based on BYOTA's ideas? We like your enthusiasm 😁
You can clone the repo and start changing BYOTA's code straight away, just follow the steps in
[Running the marimo notebook in edit mode](https://mozilla-ai.github.io/byota/getting-started/#step-2-running-the-marimo-notebook-in-edit-mode).

Anything, from alternatives to the simple dot-product re-ranker we have implemented, up to simple classifiers to be trained on top of the embeddings, sounds like an interesting idea, especially if it runs as WASM (for people to execute the code locally in their browser). You can find some
ideas for new features in the [issues section of the repo](https://github.com/mozilla-ai/byota/issues), or add new
ones if you are looking for people to collaborate with.

## 🤝 **Contributing to the Blueprint**

Want to help improve or extend this Blueprint? Check out the **[Future Features & Contributions Guide](future-features-contributions.md)** to see how you can contribute your ideas, code, or feedback to make this Blueprint even better!
