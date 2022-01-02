---
title: Algorithms
description: An example of a subfolder page.
level: one
---

# Algorithms

This is an example of a page that doesn't have a permalink defined, and
is not included in the table of contents (`_data/toc.yml`).


<div class="section-index">
    <hr class="panel-line">
    {% for file in site.pages  %}
        {% if file.level == "two" and file.cat == "algorithms" %}
            <div class="entry">
                <h5><a href="{{ file.url | prepend: site.baseurl }}">{{ file.title }}</a></h5>
                <p>{{ file.description }}</p>
            </div>
        {% endif %}
    {% endfor %}
</div>