---
layout: page
title: Blogs
permalink: /blogs/
---

# Blogs

Welcome to my website! Here you can quickly jump to a 
particular page.

<div class="section-index">
    <hr class="panel-line">
    {% for file in site.pages  %}
        {% if file.level == "one" %}
            <div class="entry">
                <h5><a href="{{ file.url | prepend: site.baseurl }}">{{ file.title }}</a></h5>
                <p>{{ file.description }}</p>
            </div>
        {% endif %}
    {% endfor %}
</div>
