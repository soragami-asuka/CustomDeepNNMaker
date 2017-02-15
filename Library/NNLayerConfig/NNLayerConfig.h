//=====================================
// ���C���[�ݒ�����쐬����֐��Q
//=====================================
#ifndef __NN_LAYER_CONFIG_H__
#define __NN_LAYER_CONFIG_H__

#ifdef NNLAYERCONFIG_EXPORTS
#define NNLAYERCONFIG_API __declspec(dllexport)
#else
#define NNLAYERCONFIG_API __declspec(dllimport)
#endif

#include"INNLayerConfig.h"
#include"INNLayerConfigEx.h"
#include"INNLayerConfigItemEx_Enum.h"

namespace CustomDeepNNLibrary
{
	/** ��̃��C���[�ݒ�����쐬���� */
	extern "C" NNLAYERCONFIG_API INNLayerConfigEx* CreateEmptyLayerConfig(const GUID& layerCode, const VersionCode& versionCode);

	/** �ݒ荀��(����)���쐬���� */
	extern "C" NNLAYERCONFIG_API INNLayerConfigItem_Float* CreateLayerCofigItem_Float(const wchar_t i_szID[], const wchar_t i_szName[], const wchar_t i_szText[], float minValue, float maxValue, float defaultValue);
	/** �ݒ荀��(����)���쐬���� */
	extern "C" NNLAYERCONFIG_API INNLayerConfigItem_Int* CreateLayerCofigItem_Int(const wchar_t i_szID[], const wchar_t i_szName[], const wchar_t i_szText[], int minValue, int maxValue, int defaultValue);
	/** �ݒ荀��(������)���쐬���� */
	extern "C" NNLAYERCONFIG_API INNLayerConfigItem_String* CreateLayerCofigItem_String(const wchar_t i_szID[], const wchar_t i_szName[], const wchar_t i_szText[], const wchar_t szDefaultValue[]);
	/** �ݒ荀��(�_���l)���쐬���� */
	extern "C" NNLAYERCONFIG_API INNLayerConfigItem_Bool* CreateLayerCofigItem_Bool(const wchar_t i_szID[], const wchar_t i_szName[], const wchar_t i_szText[], bool defaultValue);
	/** �ݒ荀��(�񋓒l)���쐬���� */
	extern "C" NNLAYERCONFIG_API INNLayerConfigItemEx_Enum* CreateLayerCofigItem_Enum(const wchar_t i_szID[], const wchar_t i_szName[], const wchar_t i_szText[]);
}


#endif // __NN_LAYERCONFIG_H__