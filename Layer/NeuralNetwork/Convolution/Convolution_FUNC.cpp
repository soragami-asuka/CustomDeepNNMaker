/*--------------------------------------------
 * FileName  : Convolution_FUNC.cpp
 * LayerName : ��݂��݃j���[�����l�b�g���[�N
 * guid      : F6662E0E-1CA4-4D59-ACCA-CAC29A16C0AA
 * 
 * Text      : ��݂��݃j���[�����l�b�g���[�N.
--------------------------------------------*/
#include"stdafx.h"

#include<string>
#include<map>

#include<Library/SettingData/Standard/SettingData.h>

#include"Convolution_FUNC.hpp"


// {F6662E0E-1CA4-4D59-ACCA-CAC29A16C0AA}
static const Gravisbell::GUID g_guid(0xf6662e0e, 0x1ca4, 0x4d59, 0xac, 0xca, 0xca, 0xc2, 0x9a, 0x16, 0xc0, 0xaa);

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
        L"��݂��݃j���[�����l�b�g���[�N",
        L"��݂��݃j���[�����l�b�g���[�N."
    };


    /** ItemData Layer Structure <id, StringData> */
    static const std::map<std::wstring, StringData> g_lpItemData_LayerStructure = 
    {
        {
            L"Output_Channel",
            {
                L"�o�̓`�����l����",
                L"�o�͂����`�����l���̐�",
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
            L"FilterSize",
            {
                L"�t�B���^�T�C�Y",
                L"��݂��݂��s�����͐M����",
            }
        },
        {
            L"Move",
            {
                L"�t�B���^�ړ���",
                L"1�j���[�������ƂɈړ�������͐M���̈ړ���",
            }
        },
        {
            L"Stride",
            {
                L"��݂��݈ړ���",
                L"��݂��݂��ƂɈړ�������͐M���̈ړ���",
            }
        },
        {
            L"PaddingM",
            {
                L"�p�f�B���O�T�C�Y(-����)",
                L"",
            }
        },
        {
            L"PaddingP",
            {
                L"�p�f�B���O�T�C�Y(+����)",
                L"",
            }
        },
        {
            L"PaddingType",
            {
                L"�p�f�B���O���",
                L"�p�f�B���O���s���ۂ̕��@�ݒ�",
            }
        },
    };


    /** ItemData Layer Structure Enum <id, enumID, StringData> */
    static const std::map<std::wstring, std::map<std::wstring, StringData>> g_lpItemDataEnum_LayerStructure =
    {
        {
            L"PaddingType",
            {
                {
                    L"zero",
                    {
                        L"�[���p�f�B���O",
                        L"�s������0�Ŗ��߂�",
                    },
                },
                {
                    L"border",
                    {
                        L"���E�l",
                        L"�s�����Ɨאڂ���l���Q�Ƃ���",
                    },
                },
                {
                    L"mirror",
                    {
                        L"���]",
                        L"�s�����Ɨאڂ���l����t�����ɎQ�Ƃ���",
                    },
                },
                {
                    L"clamp",
                    {
                        L"�N�����v",
                        L"�s�����̔��Α��̋��ڂ��珇�����ɎQ�Ƃ���",
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
	/** Name : �o�̓`�����l����
	  * ID   : Output_Channel
	  * Text : �o�͂����`�����l���̐�
	  */
	pLayerConfig->AddItem(
		Gravisbell::SettingData::Standard::CreateItem_Int(
			L"Output_Channel",
			CurrentLanguage::g_lpItemData_LayerStructure[L"Output_Channel"].name.c_str(),
			CurrentLanguage::g_lpItemData_LayerStructure[L"Output_Channel"].text.c_str(),
			1, 65535, 1));

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
			0.000000f, 1.000000f, 0.000000f));

	/** Name : �t�B���^�T�C�Y
	  * ID   : FilterSize
	  * Text : ��݂��݂��s�����͐M����
	  */
	pLayerConfig->AddItem(
		Gravisbell::SettingData::Standard::CreateItem_Vector3D_Int(
			L"FilterSize",
			CurrentLanguage::g_lpItemData_LayerStructure[L"FilterSize"].name.c_str(),
			CurrentLanguage::g_lpItemData_LayerStructure[L"FilterSize"].text.c_str(),
			1, 1, 1,
			65535, 65535, 65535,
			1, 1, 1));

	/** Name : �t�B���^�ړ���
	  * ID   : Move
	  * Text : 1�j���[�������ƂɈړ�������͐M���̈ړ���
	  */
	pLayerConfig->AddItem(
		Gravisbell::SettingData::Standard::CreateItem_Vector3D_Float(
			L"Move",
			CurrentLanguage::g_lpItemData_LayerStructure[L"Move"].name.c_str(),
			CurrentLanguage::g_lpItemData_LayerStructure[L"Move"].text.c_str(),
			0.100000f, 0.100000f, 0.100000f,
			65535.000000f, 65535.000000f, 65535.000000f,
			1.000000f, 1.000000f, 1.000000f));

	/** Name : ��݂��݈ړ���
	  * ID   : Stride
	  * Text : ��݂��݂��ƂɈړ�������͐M���̈ړ���
	  */
	pLayerConfig->AddItem(
		Gravisbell::SettingData::Standard::CreateItem_Vector3D_Float(
			L"Stride",
			CurrentLanguage::g_lpItemData_LayerStructure[L"Stride"].name.c_str(),
			CurrentLanguage::g_lpItemData_LayerStructure[L"Stride"].text.c_str(),
			0.100000f, 0.100000f, 0.100000f,
			65535.000000f, 65535.000000f, 65535.000000f,
			1.000000f, 1.000000f, 1.000000f));

	/** Name : �p�f�B���O�T�C�Y(-����)
	  * ID   : PaddingM
	  */
	pLayerConfig->AddItem(
		Gravisbell::SettingData::Standard::CreateItem_Vector3D_Int(
			L"PaddingM",
			CurrentLanguage::g_lpItemData_LayerStructure[L"PaddingM"].name.c_str(),
			CurrentLanguage::g_lpItemData_LayerStructure[L"PaddingM"].text.c_str(),
			0, 0, 0,
			65535, 65535, 65535,
			0, 0, 0));

	/** Name : �p�f�B���O�T�C�Y(+����)
	  * ID   : PaddingP
	  */
	pLayerConfig->AddItem(
		Gravisbell::SettingData::Standard::CreateItem_Vector3D_Int(
			L"PaddingP",
			CurrentLanguage::g_lpItemData_LayerStructure[L"PaddingP"].name.c_str(),
			CurrentLanguage::g_lpItemData_LayerStructure[L"PaddingP"].text.c_str(),
			0, 0, 0,
			65535, 65535, 65535,
			0, 0, 0));

	/** Name : �p�f�B���O���
	  * ID   : PaddingType
	  * Text : �p�f�B���O���s���ۂ̕��@�ݒ�
	  */
	{
		Gravisbell::SettingData::Standard::IItemEx_Enum* pItemEnum = Gravisbell::SettingData::Standard::CreateItem_Enum(
			L"PaddingType",
			CurrentLanguage::g_lpItemData_LayerStructure[L"PaddingType"].name.c_str(),
			CurrentLanguage::g_lpItemData_LayerStructure[L"PaddingType"].text.c_str());

		// 0
		pItemEnum->AddEnumItem(
			L"zero",
			L"�[���p�f�B���O",
			L"�s������0�Ŗ��߂�");
		// 1
		pItemEnum->AddEnumItem(
			L"border",
			L"���E�l",
			L"�s�����Ɨאڂ���l���Q�Ƃ���");
		// 2
		pItemEnum->AddEnumItem(
			L"mirror",
			L"���]",
			L"�s�����Ɨאڂ���l����t�����ɎQ�Ƃ���");
		// 3
		pItemEnum->AddEnumItem(
			L"clamp",
			L"�N�����v",
			L"�s�����̔��Α��̋��ڂ��珇�����ɎQ�Ƃ���");

		pLayerConfig->AddItem(pItemEnum);
	}

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
			0.000000f, 1000.000000f, 1.000000f));

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


