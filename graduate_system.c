#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h> // 日期结构体

struct Date {
    int year;
    int month;
    int day;
};


// 毕业生信息结构体
struct GraduateInfo {
    char student_id[20];        // 学号
    char name[30];              // 姓名
    char gender;                // 性别
    struct Date birth_date;     // 出生年月日
    int enrollment_year;        // 入学年份
    int graduation_year;        // 毕业年份
    char education_level[20];   // 毕业学历
    char major[50];             // 毕业专业
    char career_direction[30];  // 就业方向
    char employer[100];         // 就业/升学单位
    char job_major[50];         // 从事专业
};

// 清理输入缓冲区
void clearInputBuffer() {
    int c;
    while ((c = getchar()) != '\n' && c != EOF);
}

// 1. 批量导入就业数据
void importFromFile(struct GraduateInfo** list, int* count) {
    FILE* fp = fopen("/Users/lifeng/Documents/ai_code/student_yoyo/students.txt", "r");
    if (fp == NULL) {
        printf("\n========== 错误 ==========\n");
        printf("无法打开文件 students.txt\n");
        printf("错误信息: ");
        perror("fopen");
        printf("\n提示:\n");
        printf("1. 请确保 students.txt 文件存在于程序运行目录\n");
        printf("2. 如果使用 IDE 运行，请确保工作目录设置正确\n");
        printf("3. 如果从命令行运行，请先切换到程序所在目录:\n");
        printf("   cd /Users/lifeng/Documents/ai_code/student_yoyo\n");
        printf("   ./kese\n");
        printf("4. 或者将 students.txt 文件复制到当前工作目录\n");
        printf("==========================\n\n");
        return;
    }

    int line = 0;
    char buffer[512];

    while (fgets(buffer, sizeof(buffer), fp)) {
        line++;
        if (strlen(buffer) == 0) continue;

        // 初始化结构体，避免未初始化字段包含垃圾数据
        struct GraduateInfo new_stu;
        memset(&new_stu, 0, sizeof(struct GraduateInfo));  // 初始化为0
        
        // 简单解析，实际应根据文件格式调整
        if (sscanf(buffer, "%s %s %c %d-%d-%d %d %d %s %s %s %s %s",
            new_stu.student_id, new_stu.name, &new_stu.gender,
            &new_stu.birth_date.year, &new_stu.birth_date.month, &new_stu.birth_date.day,
            &new_stu.enrollment_year, &new_stu.graduation_year,
            new_stu.education_level, new_stu.major, new_stu.career_direction,
            new_stu.employer, new_stu.job_major) >= 12) {

            // 检查学号是否重复
            int duplicate = 0;
            for (int i = 0; i < *count; i++) {
                if (strcmp((*list)[i].student_id, new_stu.student_id) == 0) {
                    printf("毕业生学号%s已存在，跳过此毕业生！\n", new_stu.student_id);
                    duplicate = 1;
                    break;
                }
            }

            if (!duplicate) {
                // 检查是否需要扩展数组（每次扩展20个）
                // 如果list为NULL，首次分配
                if (*list == NULL) {
                    *list = (struct GraduateInfo*)malloc(20 * sizeof(struct GraduateInfo));
                    if (*list == NULL) {
                        printf("内存分配失败\n");
                        fclose(fp);
                        return;
                    }
                }
                // 如果当前count是20的倍数（且大于0），需要扩展以容纳下一个记录
                else if (*count > 0 && (*count % 20 == 0)) {
                    struct GraduateInfo* temp = (struct GraduateInfo*)realloc(*list, (*count + 20) * sizeof(struct GraduateInfo));
                    if (temp == NULL) {
                        printf("内存分配失败\n");
                        fclose(fp);
                        return;
                    }
                    *list = temp;
                }
                
                (*list)[*count] = new_stu;
                (*count)++;
                printf("成功导入学生: %s\n", new_stu.name);
            }
        }
        else {
            printf("第%d行数据格式错误\n", line);
        }
    }

    fclose(fp);

    // 保存到二进制文件
    FILE* bin_fp = fopen("/Users/lifeng/Documents/ai_code/student_yoyo/students.dat", "wb");
    if (bin_fp) {
        fwrite(*list, sizeof(struct GraduateInfo), *count, bin_fp);
        fclose(bin_fp);
        printf("数据已保存到students.dat文件\n");
    }
    else {
        printf("保存数据文件失败\n");
    }

    printf("批量导入完成，共导入%d名学生\n", *count);
}

// 2. 浏览就业数据
void displayAllGraduates(struct GraduateInfo* list, int count) {
    if (count == 0) {
        printf("当前系统无毕业生信息！\n");
        return;
    }

    printf("\n========== 所有毕业生信息 ==========\n");
    printf("%-15s %-10s %-4s %-12s %-4s %-4s %-10s %-15s %-12s %-20s %-15s\n",
        "学号", "姓名", "性别", "出生日期", "入学", "毕业", "学历", "专业",
        "就业方向", "单位", "从事专业");
    printf("----------------------------------------------------------------------------------------------------------------------------\n");

    for (int i = 0; i < count; i++) {
        printf("%-15s %-10s %-4c %04d-%02d-%02d %-4d %-4d %-10s %-15s %-12s %-20s %-15s\n",
            list[i].student_id,
            list[i].name,
            list[i].gender,
            list[i].birth_date.year,
            list[i].birth_date.month,
            list[i].birth_date.day,
            list[i].enrollment_year,
            list[i].graduation_year,
            list[i].education_level,
            list[i].major,
            list[i].career_direction,
            list[i].employer,
            list[i].job_major);
    }
    printf("共%d条记录\n", count);
}

// 3. 查询就业数据
void queryGraduate(struct GraduateInfo* list, int count, int mode) {
    if (count == 0) {
        printf("当前系统无毕业生信息！\n");
        return;
    }

    char input[100];
    printf("请输入要查询的%s: ", mode == 1 ? "学号" : "姓名");
    fgets(input, sizeof(input), stdin);
    input[strcspn(input, "\n")] = '\0';

    int found = 0;
    printf("\n========== 查询结果 ==========\n");
    printf("%-15s %-10s %-4s %-12s %-4s %-4s %-10s %-15s %-12s %-20s %-15s\n",
        "学号", "姓名", "性别", "出生日期", "入学", "毕业", "学历", "专业",
        "就业方向", "单位", "从事专业");
    printf("----------------------------------------------------------------------------------------------------------------------------\n");

    for (int i = 0; i < count; i++) {
        if ((mode == 1 && strcmp(list[i].student_id, input) == 0) ||
            (mode == 2 && strcmp(list[i].name, input) == 0)) {
            printf("%-15s %-10s %-4c %04d-%02d-%02d %-4d %-4d %-10s %-15s %-12s %-20s %-15s\n",
                list[i].student_id,
                list[i].name,
                list[i].gender,
                list[i].birth_date.year,
                list[i].birth_date.month,
                list[i].birth_date.day,
                list[i].enrollment_year,
                list[i].graduation_year,
                list[i].education_level,
                list[i].major,
                list[i].career_direction,
                list[i].employer,
                list[i].job_major);
            found++;
            }
    }

    if (found == 0) {
        printf("未找到符合条件的毕业生信息\n");
    }
    else {
        printf("共找到%d条记录\n", found);
    }
}

// 4. 增录就业数据
void addGraduate(struct GraduateInfo** list, int* count) {
    struct GraduateInfo new_stu;

    printf("\n========== 增加毕业生信息 ==========\n");

    // 输入学号并检查重复
    while (1) {
        printf("请输入学号: ");
        fgets(new_stu.student_id, sizeof(new_stu.student_id), stdin);
        new_stu.student_id[strcspn(new_stu.student_id, "\n")] = '\0';

        int duplicate = 0;
        for (int i = 0; i < *count; i++) {
            if (strcmp((*list)[i].student_id, new_stu.student_id) == 0) {
                printf("学号已存在，请重新输入\n");
                duplicate = 1;
                break;
            }
        }

        if (!duplicate) break;
    }

    printf("请输入姓名: ");
    fgets(new_stu.name, sizeof(new_stu.name), stdin);
    new_stu.name[strcspn(new_stu.name, "\n")] = '\0';

    printf("请输入性别(F/M): ");
    scanf("%c", &new_stu.gender);
    getchar();

    printf("请输入出生日期(年 月 日): ");
    scanf("%d %d %d", &new_stu.birth_date.year,
        &new_stu.birth_date.month, &new_stu.birth_date.day);
    getchar();

    printf("请输入入学年份: ");
    scanf("%d", &new_stu.enrollment_year);
    getchar();

    printf("请输入毕业年份: ");
    scanf("%d", &new_stu.graduation_year);
    getchar();  // 修复：添加 getchar() 清理缓冲区中的换行符

    printf("请输入毕业学历: ");
    fgets(new_stu.education_level, sizeof(new_stu.education_level), stdin);
    new_stu.education_level[strcspn(new_stu.education_level, "\n")] = '\0';  // 修复：将 "\\n" 改为 "\n"

    printf("请输入毕业专业: ");
    fgets(new_stu.major, sizeof(new_stu.major), stdin);
    new_stu.major[strcspn(new_stu.major, "\n")] = '\0';

    printf("请输入就业方向: ");
    fgets(new_stu.career_direction, sizeof(new_stu.career_direction), stdin);
    new_stu.career_direction[strcspn(new_stu.career_direction, "\n")] = '\0';

    printf("请输入就业/升学单位: ");
    fgets(new_stu.employer, sizeof(new_stu.employer), stdin);
    new_stu.employer[strcspn(new_stu.employer, "\n")] = '\0';

    printf("请输入从事专业: ");
    fgets(new_stu.job_major, sizeof(new_stu.job_major), stdin);
    new_stu.job_major[strcspn(new_stu.job_major, "\n")] = '\0';

    // 扩展数组
    struct GraduateInfo* temp = (struct GraduateInfo*)realloc(*list, (*count + 1) * sizeof(struct GraduateInfo));
    if (temp == NULL) {
        printf("内存分配失败\n");
        return;
    }
    *list = temp;

    // 添加新记录
    (*list)[*count] = new_stu;
    (*count)++;

    // 保存到文件
    FILE* fp = fopen("/Users/lifeng/Documents/ai_code/student_yoyo/students.dat", "wb");
    if (fp) {
        fwrite(*list, sizeof(struct GraduateInfo), *count, fp);
        fclose(fp);
        printf("数据已保存到文件\n");
    }
    else {
        printf("保存文件失败\n");
    }

    printf("新增毕业生信息成功！\n");
}


// 5. 删除就业数据
void deleteGraduate(struct GraduateInfo** list, int* count) {
    if (*count == 0) {
        printf("当前系统无毕业生信息！\n");
        return;
    }

    char input[100];
    printf("请输入要删除的学号或姓名: ");
    fgets(input, sizeof(input), stdin);
    input[strcspn(input, "\n")] = '\0';

    // 查找匹配的记录
    int found_count = 0;
    int found_indices[100]; // 假设最多100个同名

    for (int i = 0; i < *count; i++) {
        if (strcmp((*list)[i].student_id, input) == 0 ||
            strcmp((*list)[i].name, input) == 0) {
            found_indices[found_count] = i;
            found_count++;
        }
    }

    if (found_count == 0) {
        printf("未找到该毕业生\n");
        return;
    }

    // 显示找到的记录
    printf("\n找到以下毕业生:\n");
    for (int i = 0; i < found_count; i++) {
        int idx = found_indices[i];
        printf("%d. 学号:%s 姓名:%s\n", i + 1, (*list)[idx].student_id, (*list)[idx].name);
    }

    if (found_count > 1) {
        printf("找到多条记录，请输入要删除的学号: ");
        char id[20];
        fgets(id, sizeof(id), stdin);
        id[strcspn(id, "\n")] = '\0';

        // 重新查找
        int delete_index = -1;
        for (int i = 0; i < *count; i++) {
            if (strcmp((*list)[i].student_id, id) == 0) {
                delete_index = i;
                break;
            }
        }

        if (delete_index == -1) {
            printf("未找到该学号\n");
            return;
        }

        printf("确认删除学号为%s的毕业生信息？(y/n): ", id);
        char confirm;
        scanf("%c", &confirm);
        getchar();

        if (confirm == 'y' || confirm == 'Y') {
            // 删除记录
            for (int i = delete_index; i < *count - 1; i++) {
                (*list)[i] = (*list)[i + 1];
            }
            (*count)--;
            printf("删除成功！\n");
        }
        else {
            printf("取消删除\n");
        }
    }
    else {
        printf("确认删除该毕业生信息？(y/n): ");
        char confirm;
        scanf("%c", &confirm);
        getchar();

        if (confirm == 'y' || confirm == 'Y') {
            int delete_index = found_indices[0];
            for (int i = delete_index; i < *count - 1; i++) {
                (*list)[i] = (*list)[i + 1];
            }
            (*count)--;
            printf("删除成功！\n");
        }
        else {
            printf("取消删除\n");
        }
    }

    // 保存到文件
    FILE* fp = fopen("/Users/lifeng/Documents/ai_code/student_yoyo/students.dat", "wb");
    if (fp) {
        fwrite(*list, sizeof(struct GraduateInfo), *count, fp);
        fclose(fp);
        printf("数据已保存\n");
    }
}

// 6. 修改就业数据
void modifyGraduate(struct GraduateInfo* list, int count) {
    if (count == 0) {
        printf("当前系统无毕业生信息！\n");
        return;
    }

    char id[20];
    printf("请输入要修改的学号: ");
    fgets(id, sizeof(id), stdin);
    id[strcspn(id, "\n")] = '\0';

    int index = -1;
    for (int i = 0; i < count; i++) {
        if (strcmp(list[i].student_id, id) == 0) {
            index = i;
            break;
        }
    }

    if (index == -1) {
        printf("未找到该学号\n");
        return;
    }

    // 显示原信息
    printf("\n当前信息:\n");
    printf("%-15s %-10s %-4s %-12s %-4s %-4s %-10s %-15s %-12s %-20s %-15s\n",
        "学号", "姓名", "性别", "出生日期", "入学", "毕业", "学历", "专业",
        "就业方向", "单位", "从事专业");
    printf("%-15s %-10s %-4c %04d-%02d-%02d %-4d %-4d %-10s %-15s %-12s %-20s %-15s\n",
        list[index].student_id,
        list[index].name,
        list[index].gender,
        list[index].birth_date.year,
        list[index].birth_date.month,
        list[index].birth_date.day,
        list[index].enrollment_year,
        list[index].graduation_year,
        list[index].education_level,
        list[index].major,
        list[index].career_direction,
        list[index].employer,
        list[index].job_major);

    // 选择修改项
    printf("\n请选择要修改的项:\n");
    printf("1. 姓名\n");
    printf("2. 性别\n");
    printf("3. 出生日期\n");
    printf("4. 入学年份\n");
    printf("5. 毕业年份\n");
    printf("6. 毕业学历\n");
    printf("7. 毕业专业\n");
    printf("8. 就业方向\n");
    printf("9. 就业单位\n");
    printf("10. 从事专业\n");
    printf("请输入选项(1-10): ");

    int choice;
    scanf("%d", &choice);
    getchar();

    switch (choice) {
    case 1:
        printf("请输入新姓名: ");
        fgets(list[index].name, sizeof(list[index].name), stdin);
        list[index].name[strcspn(list[index].name, "\n")] = '\0';
        break;
    case 2:
        printf("请输入新性别(F/M): ");
        scanf("%c", &list[index].gender);
        getchar();
        break;
    case 3:
        printf("请输入新出生日期(年 月 日): ");
        scanf("%d %d %d", &list[index].birth_date.year,
            &list[index].birth_date.month, &list[index].birth_date.day);
        getchar();
        break;
    case 4:
        printf("请输入新入学年份: ");
        scanf("%d", &list[index].enrollment_year);
        getchar();
        break;
    case 5:
        printf("请输入新毕业年份: ");
        scanf("%d", &list[index].graduation_year);
        getchar();
        break;
    case 6:
        printf("请输入新毕业学历: ");
        fgets(list[index].education_level, sizeof(list[index].education_level), stdin);
        list[index].education_level[strcspn(list[index].education_level, "\n")] = '\0';
        break;
    case 7:
        printf("请输入新毕业专业: ");
        fgets(list[index].major, sizeof(list[index].major), stdin);
        list[index].major[strcspn(list[index].major, "\n")] = '\0';
        break;
    case 8:
        printf("请输入新就业方向: ");
        fgets(list[index].career_direction, sizeof(list[index].career_direction), stdin);
        list[index].career_direction[strcspn(list[index].career_direction, "\n")] = '\0';
        break;
    case 9:
        printf("请输入新就业单位: ");
        fgets(list[index].employer, sizeof(list[index].employer), stdin);
        list[index].employer[strcspn(list[index].employer, "\n")] = '\0';
        break;
    case 10:
        printf("请输入新从事专业: ");
        fgets(list[index].job_major, sizeof(list[index].job_major), stdin);
        list[index].job_major[strcspn(list[index].job_major, "\n")] = '\0';
        break;
    default:
        printf("无效选项\n");
        return;
    }

    // 保存到文件
    FILE* fp = fopen("/Users/lifeng/Documents/ai_code/student_yoyo/students.dat", "wb");
    if (fp) {
        fwrite(list, sizeof(struct GraduateInfo), count, fp);
        fclose(fp);
        printf("修改成功！数据已保存\n");
    }
    else {
        printf("保存文件失败\n");
    }
}

// 7. 统计某一年份就业率
void statYearEmploymentRate(struct GraduateInfo* list, int count) {
    if (count == 0) {
        printf("当前系统无毕业生信息！\n");
        return;
    }

    int year;
    printf("请输入要统计的年份: ");
    scanf("%d", &year);
    getchar();

    int total = 0;
    int employed = 0;

    for (int i = 0; i < count; i++) {
        if (list[i].graduation_year == year) {
            total++;
            // 排除"二战"、"未就业"、"其他"
            if (strcmp(list[i].career_direction, "二战") != 0 &&
                strcmp(list[i].career_direction, "未就业") != 0 &&
                strcmp(list[i].career_direction, "其他") != 0) {
                employed++;
                }
        }
    }

    if (total == 0) {
        printf("%d年没有毕业生\n", year);
    }
    else {
        float rate = (float)employed / total * 100;
        printf("\n%d年毕业生统计结果:\n", year);
        printf("毕业生总数: %d\n", total);
        printf("就业人数: %d\n", employed);
        printf("就业率: %.2f%%\n", rate);
    }
}

// 8. 统计某一年份不同学历就业率
void statEducationEmploymentRate(struct GraduateInfo* list, int count) {
    if (count == 0) {
        printf("当前系统无毕业生信息！\n");
        return;
    }

    int year;
    printf("请输入要统计的年份: ");
    scanf("%d", &year);
    getchar();

    // 初始化统计数组
    char* educations[] = { "本科生", "硕士研究生", "博士研究生" };
    int total[3] = { 0 };
    int employed[3] = { 0 };

    for (int i = 0; i < count; i++) {
        if (list[i].graduation_year == year) {
            // 统计不同学历
            for (int j = 0; j < 3; j++) {
                if (strcmp(list[i].education_level, educations[j]) == 0) {
                    total[j]++;
                    if (strcmp(list[i].career_direction, "二战") != 0 &&
                        strcmp(list[i].career_direction, "未就业") != 0 &&
                        strcmp(list[i].career_direction, "其他") != 0) {
                        employed[j]++;
                        }
                    break;
                }
            }
        }
    }

    printf("\n%d年不同学历就业率统计:\n", year);
    for (int i = 0; i < 3; i++) {
        if (total[i] > 0) {
            float rate = (float)employed[i] / total[i] * 100;
            printf("%s: 总人数=%d, 就业人数=%d, 就业率=%.2f%%\n",
                educations[i], total[i], employed[i], rate);
        }
        else {
            printf("%s: 无毕业生\n", educations[i]);
        }
    }
}

// 9. 统计某一年份不同专业就业率
void statMajorEmploymentRate(struct GraduateInfo* list, int count) {
    if (count == 0) {
        printf("当前系统无毕业生信息！\n");
        return;
    }

    int year;
    printf("请输入要统计的年份: ");
    scanf("%d", &year);
    getchar();

    // 临时存储专业统计
    char majors[100][50];
    int total[100] = { 0 };
    int employed[100] = { 0 };
    int major_count = 0;

    for (int i = 0; i < count; i++) {
        if (list[i].graduation_year == year) {
            // 查找专业是否已存在
            int found = 0;
            for (int j = 0; j < major_count; j++) {
                if (strcmp(majors[j], list[i].major) == 0) {
                    total[j]++;
                    if (strcmp(list[i].career_direction, "二战") != 0 &&
                        strcmp(list[i].career_direction, "未就业") != 0 &&
                        strcmp(list[i].career_direction, "其他") != 0) {
                        employed[j]++;
                    }
                    found = 1;
                    break;
                }
            }

            // 新专业
            if (!found) {
                strcpy(majors[major_count], list[i].major);
                total[major_count] = 1;
                if (strcmp(list[i].career_direction, "二战") != 0 &&
                    strcmp(list[i].career_direction, "未就业") != 0 &&
                    strcmp(list[i].career_direction, "其他") != 0) {
                    employed[major_count] = 1;
                }
                else {
                    employed[major_count] = 0;
                }
                major_count++;
            }
        }
    }

    printf("\n%d年不同专业就业率统计:\n", year);
    for (int i = 0; i < major_count; i++) {
        float rate = (float)employed[i] / total[i] * 100;
        printf("专业[%s]: 总人数=%d, 就业人数=%d, 就业率=%.2f%%\n",
            majors[i], total[i], employed[i], rate);
    }
}

// 10. 统计某一年份不同就业方向人数和比例
void statCareerDirection(struct GraduateInfo* list, int count) {
    if (count == 0) {
        printf("当前系统无毕业生信息！\n");
        return;
    }

    int year;
    printf("请输入要统计的年份: ");
    scanf("%d", &year);
    getchar();

    // 预设就业方向
    char* directions[] = { "直接工作", "公务员", "国内读硕", "出国读硕",
                         "国内读博", "国外读博", "二战", "二学位",
                         "未就业", "其他" };
    int direction_count[10] = { 0 };
    int total = 0;

    for (int i = 0; i < count; i++) {
        if (list[i].graduation_year == year) {
            total++;
            for (int j = 0; j < 10; j++) {
                if (strcmp(list[i].career_direction, directions[j]) == 0) {
                    direction_count[j]++;
                    break;
                }
            }
        }
    }

    if (total == 0) {
        printf("%d年没有毕业生\n", year);
        return;
    }

    printf("\n%d年不同就业方向统计:\n", year);
    printf("总毕业生数: %d\n", total);
    printf("%-12s %-8s %-8s\n", "就业方向", "人数", "比例");
    for (int i = 0; i < 10; i++) {
        if (direction_count[i] > 0) {
            float ratio = (float)direction_count[i] / total * 100;
            printf("%-12s %-8d %-7.2f%%\n", directions[i], direction_count[i], ratio);
        }
    }
}

// 11. 统计某一年份从事不同专业的人数和比例
void statJobMajor(struct GraduateInfo* list, int count) {
    if (count == 0) {
        printf("当前系统无毕业生信息！\n");
        return;
    }

    int year;
    printf("请输入要统计的年份: ");
    scanf("%d", &year);
    getchar();

    // 临时存储从事专业统计
    char job_majors[100][50];
    int major_count[100] = { 0 };
    int total = 0;
    int unique_count = 0;

    for (int i = 0; i < count; i++) {
        if (list[i].graduation_year == year &&
            strlen(list[i].job_major) > 0 &&
            strcmp(list[i].job_major, "无") != 0 &&
            strcmp(list[i].job_major, "未就业") != 0) {

            total++;

            // 查找是否已存在
            int found = 0;
            for (int j = 0; j < unique_count; j++) {
                if (strcmp(job_majors[j], list[i].job_major) == 0) {
                    major_count[j]++;
                    found = 1;
                    break;
                }
            }

            // 新专业
            if (!found) {
                strcpy(job_majors[unique_count], list[i].job_major);
                major_count[unique_count] = 1;
                unique_count++;
            }
        }
    }

    if (total == 0) {
        printf("%d年没有就业的毕业生\n", year);
        return;
    }

    printf("\n%d年从事不同专业统计:\n", year);
    printf("总就业人数: %d\n", total);
    printf("%-20s %-8s %-8s\n", "从事专业", "人数", "比例");

    for (int i = 0; i < unique_count; i++) {
        float ratio = (float)major_count[i] / total * 100;
        printf("%-20s %-8d %-7.2f%%\n", job_majors[i], major_count[i], ratio);
    }
}

// 12. 退出系统
int exitSystem(struct GraduateInfo** list, int* count) {
    printf("确认退出系统？(y/n): ");
    char confirm;
    scanf("%c", &confirm);
    getchar();

    if (confirm == 'y' || confirm == 'Y') {
        // 保存数据
        FILE* fp = fopen("/Users/lifeng/Documents/ai_code/student_yoyo/students.dat", "wb");
        if (fp) {
            fwrite(*list, sizeof(struct GraduateInfo), *count, fp);
            fclose(fp);
        }

        // 释放内存
        if (*list != NULL) {
            free(*list);
            *list = NULL;
        }

        printf("系统已退出，数据已保存\n");
        return 1;
    }
    return 0;
}

// 显示主菜单
void showMainMenu() {
    printf("\n========== 毕业生就业去向管理系统 ==========\n");
    printf("1. 批量导入就业数据\n");
    printf("2. 浏览就业数据\n");
    printf("3. 按学号查询\n");
    printf("4. 按姓名查询\n");
    printf("5. 增加就业数据\n");
    printf("6. 删除就业数据\n");
    printf("7. 修改就业数据\n");
    printf("8. 统计就业数据\n");
    printf("0. 退出系统\n");
}

// 显示统计子菜单
void showStatMenu() {
    printf("\n========== 统计功能菜单 ==========\n");
    printf("1. 统计某一年份就业率\n");
    printf("2. 统计某一年份不同学历就业率\n");
    printf("3. 统计某一年份不同专业就业率\n");
    printf("4. 统计某一年份不同就业方向人数和比例\n");
    printf("5. 统计某一年份从事不同专业的人数和比例\n");
    printf("请选择统计功能: ");
}
// 从文件加载数据
int loadFromFile(struct GraduateInfo** list, int* count) {
    FILE* fp = fopen("/Users/lifeng/Documents/ai_code/student_yoyo/students.dat", "rb");
    if (fp == NULL) {
        *list = NULL;
        *count = 0;
        return 0;
    }

    // 获取文件大小
    fseek(fp, 0, SEEK_END);
    long file_size = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    *count = file_size / sizeof(struct GraduateInfo);

    // 分配内存
    *list =(struct GraduateInfo*)malloc((*count) * sizeof(struct GraduateInfo));
    if (*list == NULL) {
        fclose(fp);
        return 0;
    }

    // 读取数据
    fread(*list, sizeof(struct GraduateInfo), *count, fp);
    fclose(fp);

    printf("从文件加载了%d条记录\n", *count);
    return 1;
}

// 主函数
int main() {
    struct GraduateInfo* graduateList = NULL;
    int count = 0;
    int choice;

    // 从文件加载现有数据
    if (loadFromFile(&graduateList, &count) == 0) {
        graduateList = NULL;
        count = 0;
    }

    while (1) {
        showMainMenu();
        printf("请输入选择(0-8): ");
        scanf("%d", &choice);
        getchar(); // 清空缓冲区

        switch (choice) {
        case 1: importFromFile(&graduateList, &count); break;
        case 2: displayAllGraduates(graduateList, count); break;
        case 3: queryGraduate(graduateList, count, 1); break; // 按学号查询
        case 4: queryGraduate(graduateList, count, 2); break; // 按姓名查询
        case 5: addGraduate(&graduateList, &count); break;
        case 6: deleteGraduate(&graduateList, &count); break;
        case 7: modifyGraduate(graduateList, count); break;
        case 8: {
            int statChoice;
            showStatMenu();
            scanf("%d", &statChoice);
            getchar();
            switch (statChoice) {
            case 1: statYearEmploymentRate(graduateList, count); break;
            case 2: statEducationEmploymentRate(graduateList, count); break;
            case 3: statMajorEmploymentRate(graduateList, count); break;
            case 4: statCareerDirection(graduateList, count); break;
            case 5: statJobMajor(graduateList, count); break;
            default: printf("无效选择\n");
            }
            break;
        }
        case 0:
            if (exitSystem(&graduateList, &count) == 1)
                return 0;
            break;
        default: printf("输入无效，请重新选择(0-8)。\n");
        }
    }

    return 0;
}
