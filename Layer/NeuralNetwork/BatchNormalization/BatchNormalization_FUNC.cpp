/*--------------------------------------------
 * FileName  : BatchNormalization_FUNC.cpp
 * LayerName : �o�b�`���K��
 * guid      : ACD11A5A-BFB5-4951-8382-1DE89DFA96A8
 * 
 * Text      : �o�b�`�P�ʂŐ��K�����s��
--------------------------------------------*/
#include"stdafx.h"

#include<string>
#include<map>

#include<Library/SettingData/Standard.h>

#include"BatchNormalization_FUNC.hpp"


// {ACD11A5A-BFB5-4951-8382-1DE89DFA96A8}
static const Gravisbell::GUID g_guid(0xacd11a5a, 0xbfb5, 0x4951, 0x83, 0x82, 0x1d, 0xe8, 0x9d, 0xfa, 0x96, 0xa8);

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
        L"�o�b�`���K��",
        L"�o�b�`�P�ʂŐ��K�����s��"
    };


    /** ItemData Layer Structure <id, StringData> */
    static const std::map<std::wstring, StringData> g_lpItemData_LayerStructure = 
    {
        {
            L"epsilon",
            {
                L"���艻�W��",
                L"���U�̒l������������ꍇ�Ɋ���Z�����肳���邽�߂̒l",
            }
        },
    };


    /** ItemData Layer Structure Enum <id, enumID, StringData> */
    static const std::map<std::wstring, std::map<std::wstring, StringData>> g_lpItemDataEnum_LayerStructure =
    {
    };



    /** ItemData Learn <id, StringData> */
    static const std::map<std::wstring, StringData> g_lpItemData_Learn = 
    {
        {
            L"Optimizer",
            {
                L"�œK�����[�`��",
                L"�d�݌덷�𔽉f�����邽�߂̃A���S���Y��.",
            }
        },
        {
            L"LearnCoeff",
            {
                L"�w�K�W��",
                L"SGD,Momentum�Ŏg�p.",
            }
        },
        {
            L"Momentum_alpha",
            {
                L"Momentum-��",
                L"Momentum�Ŏg�p.t-1�̒l�𔽉f���銄��",
            }
        },
        {
            L"AdaDelta_beta",
            {
                L"AdaDelta-��",
                L"AdaDelta�Ŏg�p.������.�����ق����������Â炢.",
            }
        },
        {
            L"AdaDelta_epsilon",
            {
                L"AdaDelta-��",
                L"AdaDelta�Ŏg�p.�⏕��.�����قǏ����l���傫���Ȃ�.",
            }
        },
        {
            L"Adam_alpha",
            {
                L"Adam-��",
                L"Adam�Ŏg�p.���Z��.�����ق����X�V����₷��.",
            }
        },
        {
            L"Adam_beta1",
            {
                L"Adam-��",
                L"Adam�Ŏg�p.������1.�����ق����������Â炢.",
            }
        },
        {
            L"Adam_beta2",
            {
                L"Adam-��",
                L"Adam�Ŏg�p.������2.�����ق����������Â炢.",
            }
        },
        {
            L"Adam_epsilon",
            {
                L"Adam-��",
                L"AdaDelta�Ŏg�p.�⏕��.�����قǏ����l���傫���Ȃ�.",
            }
        },
    };


    /** ItemData Learn Enum <id, enumID, StringData> */
    static const std::map<std::wstring, std::map<std::wstring, StringData>> g_lpItemDataEnum_Learn =
    {
        {
            L"Optimizer",
            {
                {
                    L"SGD",
                    {
                        L"SGD",
                        L"",
                    },
                },
                {
                    L"Momentum",
                    {
                        L"Momentum",
                        L"�����t�^",
                    },
                },
                {
                    L"AdaDelta",
                    {
                        L"AdaDelta",
                        L"",
                    },
                },
                {
                    L"Adam",
                    {
                        L"Adam",
                        L"",
                    },
                },
            }
        },
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
	/** Name : ���艻�W��
	  * ID   : epsilon
	  * Text : ���U�̒l������������ꍇ�Ɋ���Z�����肳���邽�߂̒l
	  */
	pLayerConfig->AddItem(
		Gravisbell::SettingData::Standard::CreateItem_Float(
			L"epsilon",
			CurrentLanguage::g_lpItemData_LayerStructure[L"epsilon"].name.c_str(),
			CurrentLanguage::g_lpItemData_LayerStructure[L"epsilon"].text.c_str(),
			0.000010f, 1.000000f, 0.000010f));

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
	/** Name : �œK�����[�`��
	  * ID   : Optimizer
	  * Text : �d�݌덷�𔽉f�����邽�߂̃A���S���Y��.
	  */
	{
		Gravisbell::SettingData::Standard::IItemEx_Enum* pItemEnum = Gravisbell::SettingData::Standard::CreateItem_Enum(
			L"Optimizer",
			CurrentLanguage::g_lpItemData_Learn[L"Optimizer"].name.c_str(),
			CurrentLanguage::g_lpItemData_Learn[L"Optimizer"].text.c_str());

		// 0
		pItemEnum->AddEnumItem(
			L"SGD",
			L"SGD",
			L"");
		// 1
		pItemEnum->AddEnumItem(
			L"Momentum",
			L"Momentum",
			L"�����t�^");
		// 2
		pItemEnum->AddEnumItem(
			L"AdaDelta",
			L"AdaDelta",
			L"");
		// 3
		pItemEnum->AddEnumItem(
			L"Adam",
			L"Adam",
			L"");

pItemEnum->SetDefaultItem(0);
pItemEnum->SetValue(pItemEnum->GetDefault());

		pLayerConfig->AddItem(pItemEnum);
	}

	/** Name : �w�K�W��
	  * ID   : LearnCoeff
	  * Text : SGD,Momentum�Ŏg�p.
	  */
	pLayerConfig->AddItem(
		Gravisbell::SettingData::Standard::CreateItem_Float(
			L"LearnCoeff",
			CurrentLanguage::g_lpItemData_Learn[L"LearnCoeff"].name.c_str(),
			CurrentLanguage::g_lpItemData_Learn[L"LearnCoeff"].text.c_str(),
			0.000000f, 1000.000000f, 1.000000f));

	/** Name : Momentum-��
	  * ID   : Momentum_alpha
	  * Text : Momentum�Ŏg�p.t-1�̒l�𔽉f���銄��
	  */
	pLayerConfig->AddItem(
		Gravisbell::SettingData::Standard::CreateItem_Float(
			L"Momentum_alpha",
			CurrentLanguage::g_lpItemData_Learn[L"Momentum_alpha"].name.c_str(),
			CurrentLanguage::g_lpItemData_Learn[L"Momentum_alpha"].text.c_str(),
			0.000000f, 1.000000f, 0.990000f));

	/** Name : AdaDelta-��
	  * ID   : AdaDelta_beta
	  * Text : AdaDelta�Ŏg�p.������.�����ق����������Â炢.
	  */
	pLayerConfig->AddItem(
		Gravisbell::SettingData::Standard::CreateItem_Float(
			L"AdaDelta_beta",
			CurrentLanguage::g_lpItemData_Learn[L"AdaDelta_beta"].name.c_str(),
			CurrentLanguage::g_lpItemData_Learn[L"AdaDelta_beta"].text.c_str(),
			0.000000f, 1.000000f, 0.950000f));

	/** Name : AdaDelta-��
	  * ID   : AdaDelta_epsilon
	  * Text : AdaDelta�Ŏg�p.�⏕��.�����قǏ����l���傫���Ȃ�.
	  */
	pLayerConfig->AddItem(
		Gravisbell::SettingData::Standard::CreateItem_Float(
			L"AdaDelta_epsilon",
			CurrentLanguage::g_lpItemData_Learn[L"AdaDelta_epsilon"].name.c_str(),
			CurrentLanguage::g_lpItemData_Learn[L"AdaDelta_epsilon"].text.c_str(),
			0.000000f, 1.000000f, 0.000001f));

	/** Name : Adam-��
	  * ID   : Adam_alpha
	  * Text : Adam�Ŏg�p.���Z��.�����ق����X�V����₷��.
	  */
	pLayerConfig->AddItem(
		Gravisbell::SettingData::Standard::CreateItem_Float(
			L"Adam_alpha",
			CurrentLanguage::g_lpItemData_Learn[L"Adam_alpha"].name.c_str(),
			CurrentLanguage::g_lpItemData_Learn[L"Adam_alpha"].text.c_str(),
			0.000000f, 1.000000f, 0.001000f));

	/** Name : Adam-��
	  * ID   : Adam_beta1
	  * Text : Adam�Ŏg�p.������1.�����ق����������Â炢.
	  */
	pLayerConfig->AddItem(
		Gravisbell::SettingData::Standard::CreateItem_Float(
			L"Adam_beta1",
			CurrentLanguage::g_lpItemData_Learn[L"Adam_beta1"].name.c_str(),
			CurrentLanguage::g_lpItemData_Learn[L"Adam_beta1"].text.c_str(),
			0.000000f, 1.000000f, 0.900000f));

	/** Name : Adam-��
	  * ID   : Adam_beta2
	  * Text : Adam�Ŏg�p.������2.�����ق����������Â炢.
	  */
	pLayerConfig->AddItem(
		Gravisbell::SettingData::Standard::CreateItem_Float(
			L"Adam_beta2",
			CurrentLanguage::g_lpItemData_Learn[L"Adam_beta2"].name.c_str(),
			CurrentLanguage::g_lpItemData_Learn[L"Adam_beta2"].text.c_str(),
			0.000000f, 1.000000f, 0.999000f));

	/** Name : Adam-��
	  * ID   : Adam_epsilon
	  * Text : AdaDelta�Ŏg�p.�⏕��.�����قǏ����l���傫���Ȃ�.
	  */
	pLayerConfig->AddItem(
		Gravisbell::SettingData::Standard::CreateItem_Float(
			L"Adam_epsilon",
			CurrentLanguage::g_lpItemData_Learn[L"Adam_epsilon"].name.c_str(),
			CurrentLanguage::g_lpItemData_Learn[L"Adam_epsilon"].text.c_str(),
			0.000000f, 1.000000f, 0.000000f));

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


