/* Generic stack
 * Author: Ruichao Jiang
 *------------------------------
 * This program implements a generic stack.
 * as a linked list. This has flexibility of size.
 
 * The data contained in an element of the linked list is a void pointer. This implements generic.
 *
 * The stack points to the beginning of the linked list, so that the pop and push operations take only O(1) time.
 *
 * Assumptions: To store the 'return' of the pop and peek function, user should dynamically allocate memory.
 *             Trying to peek or pop an empty stack will cause error and exit the program.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <assert.h>
#include <string.h>
#include <errno.h>

/* Data structure: Node
 * Not of a linked list
 *----------------------
 * next: pointer to the next node
 * data: void pointer to generic data type
 */
typedef struct node {
    struct node *next;
    void *data;
} node;

/* Data structure: Stack
 * A handle to the beginning of the linked list
 *-----------------------------
 * mem_size: sizeof the member in the linked
 * count: current number of members
 * top: pointer to the linked list.
 */
typedef struct stack {
    size_t mem_size;
    int count;
    node *top;
} stack;

//Constructor: Construct a new stack.
//User's resposibility to free after use.
stack *constructor(size_t mem_size) {
    stack *s = malloc(sizeof(stack));
    assert(s != NULL);
    s->mem_size = mem_size;
    s->count = 0;
    s->top = NULL;
    return s;
}

//Check for emptyness
bool is_Empty(stack *s) {
    return s->count == 0;
}

//Push: Add a new element at the stack top
void push(stack *s, const void *data) {
    node *new_node = malloc(sizeof(node));  // Will be freed in pop.
    assert(new_node != NULL);
    new_node->next = s->top;
    new_node->data = malloc(sizeof(s->mem_size));  // Will be freed in pop.
    assert(new_node->data != NULL);
    memcpy(new_node->data, data, s->mem_size);
    s->top = new_node;
    s->count++;
}

void pop(stack *s, void *location) {
    if(is_Empty(s)) {
        fprintf(stderr, "No more element to pop.\n", errno);
        perror("Error:");
        exit(-1);
    }else {
        node *current_node = s->top;
        memcpy(location, current_node->data, s->mem_size);
        s->top = current_node->next;
        free(current_node->data);
        free(current_node);
        s->count--;
    }
}

void peek(stack *s, void *location) {
    if(is_Empty(s)){
        fprintf(stderr, "Nothing to peek at.\n",errno);
        perror("Error:");
        exit(-1);
    }else {
        memcpy(location, s->top->data, s->mem_size);
    }
}


int main(int argc, const char * argv[]) {
    
    // Test: Stack of int
    stack *stack_int = constructor(sizeof(int));
    for(int i = 0; i < 10; ++i) {
        push(stack_int, &i);
        printf("%d was pushed in.\n", i);
    }
    void *location = malloc(sizeof(int));
    assert(location != NULL);
    for (int i = 0; i < 10; ++i) {
        peek(stack_int, location);
        printf("%d is on the top.\n", * (int *) location);
        pop(stack_int, location);
        printf("%d just popped out.\n", * (int *) location);
    }
    
    //Try to peek at and pop an empty stack
    //peek(stack_int, location);
    //pop(stack_int, location);
    
    free(location);
    free(stack_int);
    return 0;
}
