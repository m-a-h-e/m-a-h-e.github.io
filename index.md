---
tags: python,numpy,neural-network,activation-functions,loss-functions,optimizer,optimizer-algorithms,derivatives,convolution,pooling,relu,leakyrelu,softmax
---

# A Machine Learning Compendium

At the time I was studying Microelectronics and Computer Science I had the opportunity to take some fascinating machine learning lectures.
Since that time I have followed this topic up to now where the research in neural networks and machine learning gains a lot of momentum.

>Please see the <a href="#" class="show-nav-bar">navigation menu for details on machine learning and neural networks</a>, their usage for regression, prediction, and classification problems, for reinforcement learning, generative models and other interesting fields of application.

{:.caption .img}
![Machine Learning Word Cloud](assets/images/ml_word_cloud.png)

<nav id="landing-page-nav">
  <div class="toc">
    {% for cat in site.data.navigation.categories %}
    <ul class="toc-categories">
      <li>{{ cat.name }}
        <ul class="toc-cat-pages">
          {% for page_name in cat.page_names %}{% for page in site.pages %}{% if page_name == page.name %}
            <li><a href="{{ site.baseurl }}{{ page.url }}">{{ page.title }}</a></li>
          {% endif %}{% endfor %}{% endfor %}
          {% for page_name in cat.docs_names %}{% for page in site.github.public_repositories %}{% if page_name == page.name %}
            <li><a href="https://{{ site.github.owner_name }}.github.io/{{ page.name }}">{{ page.description }}</a></li>
          {% endif %}{% endfor %}{% endfor %}
          {% for page_name in cat.repo_names %}{% for page in site.github.public_repositories %}{% if page_name == page.name %}
            <li><a href="{{ page.html_url }}">{{ page.description }}</a></li>
          {% endif %}{% endfor %}{% endfor %}
          {% for page_name in cat.repo_names %}{% if page_name == "overview" %}
            <li><a href="https://github.com/{{ site.github.owner_name }}">Overview</a></li>
          {% endif %}{% endfor %}
        </ul>
      </li>
    </ul>
    {% endfor %}
  </div>
</nav>

My interest in machine learning research and the growing amount of available publications led me to the decision to - once again - dive deeper into it.

To understand all the things down to their details, I decided to implement all components of neural networks including the optimization environment from scratch using Python and the NumPy library.

So I started based on my knowledge about neural networks as they were the days I studied, combining it with the latest research outcomes regarding new activation functions, new optimizer algorithms and new network structures, altogether better suited to solve several problems.

As a nice side effect I got a better understanding of Python, NumPy, and PyTorch.

