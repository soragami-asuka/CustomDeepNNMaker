/*--------------------------------------------
 * FileName  : Activation_FUNC.cpp
 * LayerName : �������֐�
 * guid      : 99904134-83B7-4502-A0CA-728A2C9D80C7
 * 
 * Text      : �������֐�
--------------------------------------------*/
#include"stdafx.h"

#include<string>
#include<map>

#include<Library/SettingData/Standard.h>

#include"Activation_FUNC.hpp"


// {99904134-83B7-4502-A0CA-728A2C9D80C7}
static const Gravisbell::GUID g_guid(0x99904134, 0x83b7, 0x4502, 0xa0, 0xca, 0x72, 0x8a, 0x2c, 0x9d, 0x80, 0xc7);

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
        L"�������֐�",
        L"�������֐�"
    };


    /** ItemData Layer Structure <id, StringData> */
    static const std::map<std::wstring, StringData> g_lpItemData_LayerStructure = 
    {
        {
            L"ActivationType",
            {
                L"�������֐����",
                L"�g�p���銈�����֐��̎�ނ��`����",
            }
        },
        {
            L"LeakyReLU_alpha",
            {
                L"Leaky-ReLU-Alpha",
                L"Leaky-ReLU�Ŏg�p���郿�̒l",
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
                    L"LeakyReLU",
                    {
                        L"Leaky-ReLU",
                        L"y = max(alpha*x, x);\n(x=0, y=0)��ʂ�",
                    },
                },
                {
                    L"tanh",
                    {
                        L"tanh(�o�Ȑ��֐�)",
                        L"y = sin(x)/cos(x);",
                    },
                },
                {
                    L"softmax_ALL",
                    {
                        L"SoftMax�֐�",
                        L"�S�̂ɂ����鎩�g�̊�����Ԃ��֐�.\ny = e^x / ��e^x;\n",
                    },
                },
                {
                    L"softmax_ALL_crossEntropy",
                    {
                        L"SoftMax�֐�(�o�̓��C���[�p)",
                        L"�S�̂ɂ����鎩�g�̊�����Ԃ��֐�.\ny = e^x / ��e^x;\n",
                    },
                },
                {
                    L"softmax_CH",
                    {
                        L"SoftMax�֐�(CH���̂�)",
                        L"�����X,Y,Z�ɂ�����eCH�̎��g�̊�����Ԃ��֐�.\ny = e^x / ��e^x;\n",
                    },
                },
                {
                    L"softmax_CH_crossEntropy",
                    {
                        L"SoftMax�֐�(CH���̂�)(�o�̓��C���[�p)",
                        L"�����X,Y,Z�ɂ�����eCH�̎��g�̊�����Ԃ��֐�.\ny = e^x / ��e^x;\n",
                    },
                },
            }
        },
    };



    /** ItemData Learn <id, StringData> */
    static const std::map<std::wstring, StringData> g_lpItemData_Learn = 
    {
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
			L"LeakyReLU",
			L"Leaky-ReLU",
			L"y = max(alpha*x, x);\n(x=0, y=0)��ʂ�");
		// 5
		pItemEnum->AddEnumItem(
			L"tanh",
			L"tanh(�o�Ȑ��֐�)",
			L"y = sin(x)/cos(x);");
		// 6
		pItemEnum->AddEnumItem(
			L"softmax_ALL",
			L"SoftMax�֐�",
			L"�S�̂ɂ����鎩�g�̊�����Ԃ��֐�.\ny = e^x / ��e^x;\n");
		// 7
		pItemEnum->AddEnumItem(
			L"softmax_ALL_crossEntropy",
			L"SoftMax�֐�(�o�̓��C���[�p)",
			L"�S�̂ɂ����鎩�g�̊�����Ԃ��֐�.\ny = e^x / ��e^x;\n");
		// 8
		pItemEnum->AddEnumItem(
			L"softmax_CH",
			L"SoftMax�֐�(CH���̂�)",
			L"�����X,Y,Z�ɂ�����eCH�̎��g�̊�����Ԃ��֐�.\ny = e^x / ��e^x;\n");
		// 9
		pItemEnum->AddEnumItem(
			L"softmax_CH_crossEntropy",
			L"SoftMax�֐�(CH���̂�)(�o�̓��C���[�p)",
			L"�����X,Y,Z�ɂ�����eCH�̎��g�̊�����Ԃ��֐�.\ny = e^x / ��e^x;\n");

pItemEnum->SetDefaultItem(1);
pItemEnum->SetValue(pItemEnum->GetDefault());

		pLayerConfig->AddItem(pItemEnum);
	}

	/** Name : Leaky-ReLU-Alpha
	  * ID   : LeakyReLU_alpha
	  * Text : Leaky-ReLU�Ŏg�p���郿�̒l
	  */
	pLayerConfig->AddItem(
		Gravisbell::SettingData::Standard::CreateItem_Float(
			L"LeakyReLU_alpha",
			CurrentLanguage::g_lpItemData_LayerStructure[L"LeakyReLU_alpha"].name.c_str(),
			CurrentLanguage::g_lpItemData_LayerStructure[L"LeakyReLU_alpha"].text.c_str(),
			0.0000000000000000f, 0.0000000000000000f, 0.2000000029802322f));

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


