/*--------------------------------------------
 * FileName  : FullyConnect_Activation_FUNC.cpp
 * LayerName : �S�����j���[�����l�b�g���[�N���C���[(�������֐��t��)
 * guid      : BEBA34EC-C30C-4565-9386-56088981D2D7
 * 
 * Text      : �S�����j���[�����l�b�g���[�N���C���[.
 *           : �����w�Ɗ������w����̉�.
 *           : �w�K����[�w�K�W��][�h���b�v�A�E�g��]��ݒ�ł���.
--------------------------------------------*/
#include"stdafx.h"

#include<string>
#include<map>

#include<Library/SettingData/Standard.h>

#include"FullyConnect_Activation_FUNC.hpp"


// {BEBA34EC-C30C-4565-9386-56088981D2D7}
static const Gravisbell::GUID g_guid(0xbeba34ec, 0xc30c, 0x4565, 0x93, 0x86, 0x56, 0x08, 0x89, 0x81, 0xd2, 0xd7);

// VersionCode
static const Gravisbell::VersionCode g_version = {   1,   0,   0,   0}; 



struct StringData
{
    std::wstring name;
    std::wstring text;
};

namespace DefaultLanguage
{
    /** Language Code */
    static const std::wstring g_languageCode = L"ja";

    /** Base */
    static const StringData g_baseData = 
    {
        L"�S�����j���[�����l�b�g���[�N���C���[(�������֐��t��)",
        L"�S�����j���[�����l�b�g���[�N���C���[.\n�����w�Ɗ������w����̉�.\n�w�K����[�w�K�W��][�h���b�v�A�E�g��]��ݒ�ł���."
    };


    /** ItemData Layer Structure <id, StringData> */
    static const std::map<std::wstring, StringData> g_lpItemData_LayerStructure = 
    {
        {
            L"NeuronCount",
            {
                L"�j���[������",
                L"���C���[���̃j���[������.\n�o�̓o�b�t�@���ɒ�������.",
            }
        },
        {
            L"DropOut",
            {
                L"�h���b�v�A�E�g��",
                L"�O���C���[�𖳎����銄��.\n1.0�őO���C���[�̑S�o�͂𖳎�����",
            }
        },
        {
            L"ActivationType",
            {
                L"�������֐����",
                L"�g�p���銈�����֐��̎�ނ��`����",
            }
        },
        {
            L"BoolSample",
            {
                L"Bool�^�̃T���v��",
                L"",
            }
        },
        {
            L"StringSample",
            {
                L"String�^�̃T���v��",
                L"",
            }
        },
    };


    /** ItemData Layer Structure Enum <id, enumID, StringData> */
    static const std::map<std::wstring, std::map<std::wstring, StringData>> g_lpItemDataEnum_LayerStructure =
    {
        {
            L"ActivationType",
            {
                {
                    L"lenear",
                    {
                        L"���j�A�֐�",
                        L"y = x;",
                    },
                },
                {
                    L"sigmoid",
                    {
                        L"�V�O���C�h�֐�",
                        L"y = 1 / (1 + e^(-x));\n�͈� 0 < y < 1\n(x=0, y=0.5)��ʂ�",
                    },
                },
                {
                    L"sigmoid_crossEntropy",
                    {
                        L"�V�O���C�h�֐�(�o�̓��C���[�p)",
                        L"y = 1 / (1 + e^(-x));\n�͈� 0 < y < 1\n(x=0, y=0.5)��ʂ�",
                    },
                },
                {
                    L"ReLU",
                    {
                        L"ReLU�i�����v�֐��j",
                        L"y = max(0, x);\n�͈� 0 <= y\n(x=0, y=0)��ʂ�",
                    },
                },
                {
                    L"softmax",
                    {
                        L"SoftMax�֐�",
                        L"�S�̂ɂ����鎩�g�̊�����Ԃ��֐�.\ny = e^x / ��e^x;\n",
                    },
                },
                {
                    L"softmax_crossEntropy",
                    {
                        L"SoftMax�֐�(�o�̓��C���[�p)",
                        L"�S�̂ɂ����鎩�g�̊�����Ԃ��֐�.\ny = e^x / ��e^x;\n",
                    },
                },
            }
        },
    };



    /** ItemData Learn <id, StringData> */
    static const std::map<std::wstring, StringData> g_lpItemData_Learn = 
    {
        {
            L"LearnCoeff",
            {
                L"�w�K�W��",
                L"",
            }
        },
    };


    /** ItemData Learn Enum <id, enumID, StringData> */
    static const std::map<std::wstring, std::map<std::wstring, StringData>> g_lpItemDataEnum_Learn =
    {
    };


}


namespace CurrentLanguage
{
    /** Language Code */
    static const std::wstring g_languageCode = DefaultLanguage::g_languageCode;


    /** Base */
    static StringData g_baseData = DefaultLanguage::g_baseData;


    /** ItemData Layer Structure <id, StringData> */
    static std::map<std::wstring, StringData> g_lpItemData_LayerStructure = DefaultLanguage::g_lpItemData_LayerStructure;

    /** ItemData Learn Enum <id, enumID, StringData> */
    static const std::map<std::wstring, std::map<std::wstring, StringData>> g_lpItemDataEnum_LayerStructure = DefaultLanguage::g_lpItemDataEnum_LayerStructure;


    /** ItemData Learn <id, StringData> */
    static std::map<std::wstring, StringData> g_lpItemData_Learn = DefaultLanguage::g_lpItemData_Learn;

    /** ItemData Learn Enum <id, enumID, StringData> */
    static const std::map<std::wstring, std::map<std::wstring, StringData>> g_lpItemDataEnum_Learn = DefaultLanguage::g_lpItemDataEnum_Learn;

}



/** Acquire the layer identification code.
  * @param  o_layerCode    Storage destination buffer.
  * @return On success 0. 
  */
EXPORT_API Gravisbell::ErrorCode GetLayerCode(Gravisbell::GUID& o_layerCode)
{
	o_layerCode = g_guid;

	return Gravisbell::ERROR_CODE_NONE;
}

/** Get version code.
  * @param  o_versionCode    Storage destination buffer.
  * @return On success 0. 
  */
EXPORT_API Gravisbell::ErrorCode GetVersionCode(Gravisbell::VersionCode& o_versionCode)
{
	o_versionCode = g_version;

	return Gravisbell::ERROR_CODE_NONE;
}


/** Create a layer structure setting.
  * @return If successful, new configuration information.
  */
EXPORT_API Gravisbell::SettingData::Standard::IData* CreateLayerStructureSetting(void)
{
	Gravisbell::GUID layerCode;
	GetLayerCode(layerCode);

	Gravisbell::VersionCode versionCode;
	GetVersionCode(versionCode);


	// Create Empty Setting Data
	Gravisbell::SettingData::Standard::IDataEx* pLayerConfig = Gravisbell::SettingData::Standard::CreateEmptyData(layerCode, versionCode);
	if(pLayerConfig == NULL)
		return NULL;


	// Create Item
	/** Name : �j���[������
	  * ID   : NeuronCount
	  * Text : ���C���[���̃j���[������.
	  *      : �o�̓o�b�t�@���ɒ�������.
	  */
	pLayerConfig->AddItem(
		Gravisbell::SettingData::Standard::CreateItem_Int(
			L"NeuronCount",
			CurrentLanguage::g_lpItemData_LayerStructure[L"NeuronCount"].name.c_str(),
			CurrentLanguage::g_lpItemData_LayerStructure[L"NeuronCount"].text.c_str(),
			1, 65535, 200));

	/** Name : �h���b�v�A�E�g��
	  * ID   : DropOut
	  * Text : �O���C���[�𖳎����銄��.
	  *      : 1.0�őO���C���[�̑S�o�͂𖳎�����
	  */
	pLayerConfig->AddItem(
		Gravisbell::SettingData::Standard::CreateItem_Float(
			L"DropOut",
			CurrentLanguage::g_lpItemData_LayerStructure[L"DropOut"].name.c_str(),
			CurrentLanguage::g_lpItemData_LayerStructure[L"DropOut"].text.c_str(),
			0.0000000000000000f, 1.0000000000000000f, 0.0000000000000000f));

	/** Name : �������֐����
	  * ID   : ActivationType
	  * Text : �g�p���銈�����֐��̎�ނ��`����
	  */
	{
		Gravisbell::SettingData::Standard::IItemEx_Enum* pItemEnum = Gravisbell::SettingData::Standard::CreateItem_Enum(
			L"ActivationType",
			CurrentLanguage::g_lpItemData_LayerStructure[L"ActivationType"].name.c_str(),
			CurrentLanguage::g_lpItemData_LayerStructure[L"ActivationType"].text.c_str());

		// 0
		pItemEnum->AddEnumItem(
			L"lenear",
			L"���j�A�֐�",
			L"y = x;");
		// 1
		pItemEnum->AddEnumItem(
			L"sigmoid",
			L"�V�O���C�h�֐�",
			L"y = 1 / (1 + e^(-x));\n�͈� 0 < y < 1\n(x=0, y=0.5)��ʂ�");
		// 2
		pItemEnum->AddEnumItem(
			L"sigmoid_crossEntropy",
			L"�V�O���C�h�֐�(�o�̓��C���[�p)",
			L"y = 1 / (1 + e^(-x));\n�͈� 0 < y < 1\n(x=0, y=0.5)��ʂ�");
		// 3
		pItemEnum->AddEnumItem(
			L"ReLU",
			L"ReLU�i�����v�֐��j",
			L"y = max(0, x);\n�͈� 0 <= y\n(x=0, y=0)��ʂ�");
		// 4
		pItemEnum->AddEnumItem(
			L"softmax",
			L"SoftMax�֐�",
			L"�S�̂ɂ����鎩�g�̊�����Ԃ��֐�.\ny = e^x / ��e^x;\n");
		// 5
		pItemEnum->AddEnumItem(
			L"softmax_crossEntropy",
			L"SoftMax�֐�(�o�̓��C���[�p)",
			L"�S�̂ɂ����鎩�g�̊�����Ԃ��֐�.\ny = e^x / ��e^x;\n");

pItemEnum->SetDefaultItem(1);
pItemEnum->SetValue(pItemEnum->GetDefault());

		pLayerConfig->AddItem(pItemEnum);
	}

	/** Name : Bool�^�̃T���v��
	  * ID   : BoolSample
	  */
	pLayerConfig->AddItem(
		Gravisbell::SettingData::Standard::CreateItem_Bool(
			L"BoolSample",
			CurrentLanguage::g_lpItemData_LayerStructure[L"BoolSample"].name.c_str(),
			CurrentLanguage::g_lpItemData_LayerStructure[L"BoolSample"].text.c_str(),
			true));

	/** Name : String�^�̃T���v��
	  * ID   : StringSample
	  */
	pLayerConfig->AddItem(
		Gravisbell::SettingData::Standard::CreateItem_String(
			L"StringSample",
			CurrentLanguage::g_lpItemData_LayerStructure[L"StringSample"].name.c_str(),
			CurrentLanguage::g_lpItemData_LayerStructure[L"StringSample"].text.c_str(),
			L"�T���v��"));

	return pLayerConfig;
}

/** Create layer structure settings from buffer.
  * @param  i_lpBuffer       Start address of the read buffer.
  * @param  i_bufferSize     The size of the readable buffer.
  * @param  o_useBufferSize  Buffer size actually read.
  * @return If successful, the configuration information created from the buffer
  */
EXPORT_API Gravisbell::SettingData::Standard::IData* CreateLayerStructureSettingFromBuffer(const BYTE* i_lpBuffer, int i_bufferSize, int& o_useBufferSize)
{
	Gravisbell::SettingData::Standard::IDataEx* pLayerConfig = (Gravisbell::SettingData::Standard::IDataEx*)CreateLayerStructureSetting();
	if(pLayerConfig == NULL)
		return NULL;

	int useBufferSize = pLayerConfig->ReadFromBuffer(i_lpBuffer, i_bufferSize);
	if(useBufferSize < 0)
	{
		delete pLayerConfig;

		return NULL;
	}
	o_useBufferSize = useBufferSize;

	return pLayerConfig;
}


/** Create a learning setting.
  * @return If successful, new configuration information. */
EXPORT_API Gravisbell::SettingData::Standard::IData* CreateLearningSetting(void)
{
	Gravisbell::GUID layerCode;
	GetLayerCode(layerCode);

	Gravisbell::VersionCode versionCode;
	GetVersionCode(versionCode);


	// Create Empty Setting Data
	Gravisbell::SettingData::Standard::IDataEx* pLayerConfig = Gravisbell::SettingData::Standard::CreateEmptyData(layerCode, versionCode);
	if(pLayerConfig == NULL)
		return NULL;


	// Create Item
	/** Name : �w�K�W��
	  * ID   : LearnCoeff
	  */
	pLayerConfig->AddItem(
		Gravisbell::SettingData::Standard::CreateItem_Float(
			L"LearnCoeff",
			CurrentLanguage::g_lpItemData_Learn[L"LearnCoeff"].name.c_str(),
			CurrentLanguage::g_lpItemData_Learn[L"LearnCoeff"].text.c_str(),
			0.0000000000000000f, 1000.0000000000000000f, 1.0000000000000000f));

	return pLayerConfig;
}

/** Create learning settings from buffer.
  * @param  i_lpBuffer       Start address of the read buffer.
  * @param  i_bufferSize     The size of the readable buffer.
  * @param  o_useBufferSize  Buffer size actually read.
  * @return If successful, the configuration information created from the buffer
  */
EXPORT_API Gravisbell::SettingData::Standard::IData* CreateLearningSettingFromBuffer(const BYTE* i_lpBuffer, int i_bufferSize, int& o_useBufferSize)
{
	Gravisbell::SettingData::Standard::IDataEx* pLayerConfig = (Gravisbell::SettingData::Standard::IDataEx*)CreateLearningSetting();
	if(pLayerConfig == NULL)
		return NULL;

	int useBufferSize = pLayerConfig->ReadFromBuffer(i_lpBuffer, i_bufferSize);
	if(useBufferSize < 0)
	{
		delete pLayerConfig;

		return NULL;
	}
	o_useBufferSize = useBufferSize;

	return pLayerConfig;
}


