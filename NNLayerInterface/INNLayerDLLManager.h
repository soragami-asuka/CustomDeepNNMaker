//=======================================
// ���C���[DLL�N���X
//=======================================
#ifndef __I_NN_LAYER_DLL_MANAGER_H__
#define __I_NN_LAYER_DLL_MANAGER_H__

#include<guiddef.h>

#include"INNLayerDLL.h"

namespace CustomDeepNNLibrary
{
	class INNLayerDLLManager
	{
	public:
		/** �R���X�g���N�^ */
		INNLayerDLLManager(){}
		/** �f�X�g���N�^ */
		virtual ~INNLayerDLLManager(){}

	public:
		/** DLL��ǂݍ���ŁA�Ǘ��ɒǉ�����.
			@param szFilePath		�ǂݍ��ރt�@�C���̃p�X.
			@param o_addLayerCode	�ǉ����ꂽGUID�̊i�[��A�h���X.
			@return	���������ꍇ0���Ԃ�. */
		virtual ELayerErrorCode ReadLayerDLL(const wchar_t szFilePath[], GUID& o_addLayerCode) = 0;
		/** DLL��ǂݍ���ŁA�Ǘ��ɒǉ�����.
			@param szFilePath	�ǂݍ��ރt�@�C���̃p�X.
			@return	���������ꍇ0���Ԃ�. */
		virtual ELayerErrorCode ReadLayerDLL(const wchar_t szFilePath[]) = 0;

		/** �Ǘ����Ă��郌�C���[DLL�̐����擾���� */
		virtual unsigned int GetLayerDLLCount()const = 0;
		/** �Ǘ����Ă��郌�C���[DLL��ԍ��w��Ŏ擾����.
			@param	num	�擾����DLL�̊Ǘ��ԍ�.
			@return ���������ꍇ��DLL�N���X�̃A�h���X. ���s�����ꍇ��NULL */
		virtual const INNLayerDLL* GetLayerDLLByNum(unsigned int num)const = 0;
		/** �Ǘ����Ă��郌�C���[DLL��guid�w��Ŏ擾����.
			@param guid	�擾����DLL��GUID.
			@return ���������ꍇ��DLL�N���X�̃A�h���X. ���s�����ꍇ��NULL */
		virtual const INNLayerDLL* GetLayerDLLByGUID(GUID i_layerCode)const = 0;

		/** ���C���[DLL���폜����. */
		virtual ELayerErrorCode EraseLayerDLL(GUID i_layerCode) = 0;
	};
}

#endif