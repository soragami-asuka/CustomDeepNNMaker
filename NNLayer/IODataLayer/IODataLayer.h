#ifdef IODataLayer_EXPORTS
#define IODataLayer_API __declspec(dllexport)
#else
#define IODataLayer_API __declspec(dllimport)
#endif

#include"IIODataLayer.h"


/** ���͐M���f�[�^���C���[���쐬����.GUID�͎������蓖��.CPU����
	@param bufferSize	�o�b�t�@�̃T�C�Y.��float�^�z��̗v�f��.
	@return	���͐M���f�[�^���C���[�̃A�h���X */
extern "C" IODataLayer_API CustomDeepNNLibrary::IIODataLayer* CreateIODataLayerCPU(CustomDeepNNLibrary::IODataStruct ioDataStruct);
/** ���͐M���f�[�^���C���[���쐬����.CPU����
	@param guid			���C���[��GUID.
	@param bufferSize	�o�b�t�@�̃T�C�Y.��float�^�z��̗v�f��.
	@return	���͐M���f�[�^���C���[�̃A�h���X */
extern "C" IODataLayer_API CustomDeepNNLibrary::IIODataLayer* CreateIODataLayerCPUwithGUID(GUID guid, CustomDeepNNLibrary::IODataStruct ioDataStruct);

/** ���͐M���f�[�^���C���[���쐬����.GPU����
	@param guid			���C���[��GUID.
	@param bufferSize	�o�b�t�@�̃T�C�Y.��float�^�z��̗v�f��.
	@return	���͐M���f�[�^���C���[�̃A�h���X */
extern "C" IODataLayer_API CustomDeepNNLibrary::IIODataLayer* CreateIODataLayerGPU(GUID guid, CustomDeepNNLibrary::IODataStruct ioDataStruct);
