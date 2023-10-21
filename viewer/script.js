// Example Data
const data = {
    mysteries: ['Mystery 1', 'Mystery 2'],
    placements: ['Placement 1', 'Placement 2'],
    allocation: ['Team 1', 'Team 2']
}

const loaded = {
    mysteries: false,
    placements: false,
    allocation: false
}

var selected_item_elem = null;

function t(q, n) {
    var innerhtml = '<ul>';

    n.children.forEach((child) => {
        innerhtml += `<li><p>${child.value}</p><div class="children"></div></li>`
    })

    innerhtml += '</ul>';
    q.innerHTML = innerhtml;

    q.querySelectorAll('li').forEach((elem, cidx) => {
        elem.addEventListener('click', function(e) {
            e.preventDefault()
            e.stopPropagation();
            const childrenDiv = elem.querySelector('.children');
            childrenDiv.style.display = childrenDiv.style.display == 'block' ? 'none' : 'block';
            t(childrenDiv, n.children[cidx])
        })
    })
}

function treant_structure(tree){
    var ntree = {}
    ntree.text = {'name': tree.value}
    ntree.value = tree.value
    ntree.collapsed = tree.children.length > 0 ? false : false
    ntree.collapsable = false //tree.children.length > 0
    ntree.stackChildren = true
    var updated_children = []
    tree.children.forEach(c => {
        updated_children.push(treant_structure(c))
    })
    ntree.children = updated_children
    return ntree
}

function treevis_structure(tree){
    var ntree = {}
    ntree.name = tree.value
    ntree.open = false
    ntree.type = tree.children.length > 0 ? Tree.FOLDER : Tree.FILE
    var updated_children = []
    tree.children.forEach(c => {
        updated_children.push(treevis_structure(c))
    })
    ntree.children = updated_children
    return ntree
}

function tree_struct_for_tree_vis(tree, nidx,  lst){
    const father_idx = nidx == null ? 1 : nidx;
    var node_idx = nidx == null ? 1 : nidx;

    if (nidx == null){
        lst.push({
            'id': 1,
            'father': null,
            'text_1': tree.value
        })
    }

    tree.children.forEach((child, cidx) => {
        node_idx = node_idx + 1;
        lst.push({
            'id': node_idx,
            'father': father_idx,
            'text_1': child.value
        })

        var info = tree_struct_for_tree_vis(child, node_idx, lst)
        lst = info.lst
        node_idx = info.node_idx
    })

    return {'lst': lst, 'node_idx': node_idx}
}

function category_click(category){
    try{
        const itemsDiv = category.querySelector('.items');
        itemsDiv.style.display = itemsDiv.style.display == 'block' ? 'none' : 'block';

        const categoryType = category.getAttribute('data-category');

        if (categoryType == 'custom'){
            return
        }

        if (loaded[categoryType]){
            return
        }

        loaded[categoryType] = true

        const items = data[categoryType];
        let itemsHTML = '';
        items.forEach((item, idx) => {
            itemsHTML += `<div class="item" data-category=${category} data-idx=${idx}>(${idx+1}): `
            if (categoryType == 'mysteries'){
                itemsHTML += `Who killed ${item.questions[0].intermediate_data[0].victim_info.victim}?`;
            }else if(categoryType == 'placements'){
                itemsHTML += `${item.context.split('.')[0]}.`
            }else if(categoryType == 'allocation'){
                itemsHTML += `${item.context.split('.')[0]}.`
            }else {
                itemsHTML += `${item.context.split('.')[0]}.`
            }
            itemsHTML += '</div>'
        });
        itemsDiv.innerHTML = itemsHTML + '';

        const item_divs = category.querySelectorAll('.item');
        item_divs.forEach(i => {
            i.addEventListener('click', e => {
                e.preventDefault();
                e.stopPropagation();

                const item_info = items[i.getAttribute('data-idx')]

                var ctx = '<p>' + item_info.context.split('\n').join('</p><p>') + '</p>';

                var question_HTML = '<p></p><p></p><span class="questions_header">Questions:</span><ol>'
                item_info.questions.forEach(question => {
                    question_HTML += `<li><p>${question.question}</p><p>Choices:<ul>`
                    question.choices.forEach((c, ci) => {
                        question_HTML += `<li><span class="${ci.toString() == question.answer ? 'answer' : ''}">${c}</span></li>`
                    })
                    question_HTML += '</ul></p></li>'
                })
                question_HTML += '</ol>'

                ctx += question_HTML

                document.querySelector('.story').innerHTML = ctx;
                // document.querySelector('.questions').innerHTML = question_HTML

                var tree_info = {'value': 'ROOT', children: []}
                item_info.questions[0].intermediate_trees.forEach(tree => {
                    const tree_data = tree_struct_for_tree_vis(tree.nodes[0], null, [])
                    console.log(tree_data);
                    tree_info.children.push(tree.nodes[0])
                })

                tree_info = treevis_structure(tree_info)

                document.getElementById('treevis').innerHTML = '';
                var tree = new Tree(document.getElementById('treevis'));
                tree.json([tree_info])

                onStorySelected();
                onTreeSelected();

                if (selected_item_elem != null){
                    selected_item_elem.classList.remove('selected')
                }
                selected_item_elem = i;
                selected_item_elem.classList.add('selected')

                // document.querySelector('.selected-title').innerHTML = `<b>${i.parentElement.parentElement.querySelector('h2').textContent}</b><p>${i.textContent}</p>`



            })
        })
    } catch {
        const categoryType = category.getAttribute('data-category');
        loaded[categoryType] = false
    }

}
document.addEventListener('DOMContentLoaded', function () {
    const categories = document.querySelectorAll('.category');
    categories.forEach(category => {
        category.addEventListener('click', function () {
            category_click(category)
        });
    });
});

function toggleFullscreen(panel) {
    const allPanels = document.querySelectorAll('.panel');
    const mainContainer = document.querySelector('.container'); // assuming .container is the main div

    if (panel.classList.contains('fullscreen')) {
        panel.querySelector('.expand-btn').innerHTML = panel.querySelector('.expand-btn').innerHTML.replace('Collapse', 'Expand')


        panel.classList.remove('fullscreen');
        // panel.querySelector('.close-btn').remove();

        // when exiting fullscreen, we want to show all other panels
        allPanels.forEach(p => p.style.visibility = 'visible');
        mainContainer.style.pointerEvents = 'auto'; // re-enable interactions with other elements
    } else {
        panel.classList.add('fullscreen');

        panel.querySelector('.expand-btn').innerHTML = panel.querySelector('.expand-btn').innerHTML.replace('Expand', 'Collapse')

        // when a panel is in fullscreen, we want to hide all other panels
        allPanels.forEach(p => {
            if (p !== panel) { // do not hide the panel that's in fullscreen mode
                p.style.visibility = 'hidden';
            }
        });
        mainContainer.style.pointerEvents = 'none'; // disable interactions with other elements
        panel.style.pointerEvents = 'auto'; // keep the fullscreen panel interactive
    }
}

document.addEventListener('DOMContentLoaded', function () {

    // Add the click event to all 'expand' buttons
    let expandButtons = document.querySelectorAll('.expand-btn');
    expandButtons.forEach(function (btn) {
        btn.addEventListener('click', function () {
            let panel = btn.parentElement;
            toggleFullscreen(panel);
        });
    });
});

const toggleSidebarBtn = document.getElementById('toggle-sidebar');
const sidebar = document.querySelector('.sidebar');
const content = document.querySelector('.content')

toggleSidebarBtn.addEventListener('click', () => {
    if (sidebar.classList.contains('collapsed')) {
        sidebar.classList.remove('collapsed');
        sidebar.classList.add('expand');
        toggleSidebarBtn.classList.remove('collapsed');
        content.classList.remove('sidebarcollapsed');
    } else {
        sidebar.classList.remove('expand');
        sidebar.classList.add('collapsed');
        toggleSidebarBtn.classList.add('collapsed');
        content.classList.add('sidebarcollapsed');
    }

});

const storyBtn = document.getElementById('story-btn');
const treeBtn = document.getElementById('tree-btn');

// Assuming you have a function that detects a story selection:
function onStorySelected() {
    storyBtn.removeAttribute('disabled');
}

function onTreeSelected() {
    treeBtn.removeAttribute('disabled');
}

storyBtn.addEventListener('click', () => {
    var tree_panel = document.querySelector('.tree_panel')
    var story_panel = document.querySelector('.story_panel')

    if (tree_panel.classList.contains('fullscreen')){
        toggleFullscreen(tree_panel)
    }

    toggleFullscreen(story_panel)
});

treeBtn.addEventListener('click', () => {
    var tree_panel = document.querySelector('.tree_panel')
    var story_panel = document.querySelector('.story_panel')

    if (story_panel.classList.contains('fullscreen')){
        toggleFullscreen(story_panel)
    }

    toggleFullscreen(tree_panel)

});



document.getElementById("madeYourOwnButton").addEventListener("click", function() {
    document.getElementById("uploadModal").style.display = "block";
});

document.getElementById('uploadModal').addEventListener('click', function() {
    if (document.getElementById("uploadModal").style.display == 'block'){
          document.getElementById("uploadModal").style.display = "none";
    }
})

document.getElementById('modal_content').addEventListener('click', function(e) {
    e.stopPropagation();
    // e.preventDefault();
})

document.getElementById("submitButton").addEventListener("click", function() {
    var fileInput = document.getElementById("fileInput");
    var datasetTitle = document.getElementById("datasetTitle").value;

    if (fileInput.files.length > 0) {
        var file = fileInput.files[0];
        var reader = new FileReader();

        reader.onload = function(e) {
            var content = e.target.result;
            var jsonData = JSON.parse(content);
            // Do something with jsonData and datasetTitle

            data[datasetTitle] = jsonData;

            const sidebar = document.querySelector('.sidebar-content');
            const button_cat = document.createElement('div')
            button_cat.classList.add('category')
            button_cat.setAttribute('data-category', datasetTitle)
            var button_header = document.createElement('h2');
            button_header.innerHTML = datasetTitle;
            var button_items = document.createElement('div')
            button_items.classList.add('items')

            button_cat.append(button_header);
            button_cat.append(button_items);

            const targetPosition = sidebar.children[sidebar.children.length - 1];
            sidebar.insertBefore(button_cat, targetPosition);

            button_cat.addEventListener('click', function () {
                category_click(button_cat)
            });
        }

        reader.readAsText(file);
    } else {
        alert("Please select a JSON file.");
    }

    document.getElementById("uploadModal").style.display = "none";

});


// Load data from JSON file
fetch('datasets/murder_mystery.json')
    .then(response => response.json())
    .then(jsonData => {
        data.mysteries = jsonData;
    });

fetch('datasets/object_placements.json')
    .then(response => response.json())
    .then(jsonData => {
        data.placements = jsonData;
    });

fetch('datasets/team_allocation.json')
    .then(response => response.json())
    .then(jsonData => {
        data.allocation = jsonData;
    });