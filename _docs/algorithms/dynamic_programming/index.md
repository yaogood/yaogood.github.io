---
title: Dynamic Programming
level: one
description: An example of a subfolder page.
---


# Dynamic Programming

动态规划问题包含的几点重要因素




### 经典动态规划问题

<div class="section-index">
    <hr class="panel-line">
    {% for post in site.docs.algorithms.dynamic_programming %}  
    <div class="entry">
    <h5><a href="{{ post.url | prepend: site.baseurl }}">{{ post.title }}</a></h5>
    <p>{{ post.description }}</p>
    </div>{% endfor %}
</div>
