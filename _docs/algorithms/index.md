---
title: Algorithms
level: one
description: An example of a subfolder page.
---

# Algorithms

This is an example of a page that doesn't have a permalink defined, and
is not included in the table of contents (`_data/toc.yml`).



<div class="section-index">
    <hr class="panel-line">
    {% for post in site.docs.algorithms  %}        
    <div class="entry">
    <h5><a href="{{ post.url | prepend: site.baseurl }}">{{ post.title }}</a></h5>
    <p>{{ post.description }}</p>
    </div>{% endfor %}
</div>